import numpy as np
import torch
from torch import nn
import dgl
import dgl.data
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class RGCN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = dglnn.HeteroLinear({
            'Gcell': dim, 'Gnet':dim}, dim)
        self.conv1 = dglnn.HeteroGraphConv({
            'cc': dglnn.GraphConv(dim, 2*dim),
            'cn': dglnn.GraphConv(dim, 2*dim),
            'nc': dglnn.GraphConv(dim, 2*dim),
            'nn': dglnn.GraphConv(dim, 2*dim)
        })
        self.lin2 = dglnn.HeteroLinear({
            'Gcell': 2*dim, 'Gnet': 2*dim}, dim)
    def forward(self, graph, h):
        h1 = self.lin1(h)
        h1 = {k:F.gelu(v) for k, v in h1.items()}
        h1 = self.conv1(graph, h1)
        h1 = {k:F.gelu(v) for k, v in h1.items()}
        h1 = self.lin2(h1)
        h1 = {k:F.gelu(v) for k, v in h1.items()}
        h['Gcell'] = h['Gcell'] + h1['Gcell']
        h['Gnet'] = h['Gnet'] + h1['Gnet']
        return h

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.,image_height=256,image_width=256,patch_height=4, patch_width=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.g2p = Rearrange('(b n m) f -> b (n m) f', n=image_height//patch_height, m=image_width//patch_width)#batch_graph --> pixel
        self.p2g = Rearrange('b (n m) f -> (b n m) f', n=image_height//patch_height, m=image_width//patch_width)#pixel --> batch_graph
        for _ in range(depth-1):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
                RGCN(dim)
            ]))
        self.last_layer = nn.ModuleList([])
        self.last_layer.append(nn.ModuleList([
            PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
        ]))

    def forward(self, g, h):
        # record xs, by Peng
        xs = []

        for attn, ff, rgcn in self.layers:
            x = self.g2p(h['Gcell'])
            # record xs, by Peng
            xs.append(x)

            x = attn(x) + x
            x = ff(x) + x
            x = self.p2g(x)

            sg = dgl.sampling.sample_neighbors(g, {'Gcell': range(g.num_nodes('Gcell')), 'Gnet': range(g.num_nodes('Gnet'))}, 4)
            h = rgcn(sg, h)

        for atten, ff in self.last_layer:
            x = self.g2p(h['Gcell'])
            # record xs, by Peng
            xs.append(x)

            x = attn(x) + x
            x = ff(x) + x
        return x, xs

class ViG(nn.Module):
    def __init__(self,
                *,
                image_size,
                patch_size,
                num_classes,
                dim,
                depth,
                heads,
                mlp_dim,
                pool = 'cls',
                channels = 3,
                dim_head = 64,
                dropout = 0.,
                emb_dropout = 0.):
        super().__init__()

        self.collecting = False

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.embedding = dglnn.HeteroLinear({'Gcell': channels*patch_height*patch_width, 'Gnet':3}, dim)

        self.to_pixel = Rearrange('b (h w) c -> b c h w', h=image_height // patch_height, w=image_width // patch_width)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, image_height, image_width, patch_height, patch_width)
        self.g2p = Rearrange('(b n m) f -> b (n m) f', n=image_height//patch_height, m=image_width//patch_width) # batch_graph --> pixel
        self.p2g = Rearrange('b (n m) f -> (b n m) f', n=image_height//patch_height, m=image_width//patch_width) # pixel --> batch_graph

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, dim//4, 2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(dim//4),
            nn.ConvTranspose2d(dim//4, dim//4, 2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(dim//4),
            nn.ConvTranspose2d(dim//4, 2, 2, 2),
        )

    def forward(self, g):
        h = g.ndata['h']
        h = self.embedding(h)

        x = self.g2p(h['Gcell'])
        # record xs, by Peng
        self.record(x)

        b, n, _ = x.shape
        x += self.pos_embedding

        x = self.dropout(x)
        x = self.p2g(x)
        h['Gcell'] = x
        x, xs = self.transformer(g, h)

        # record xs, by Peng
        for xs_i in xs:
            self.record(xs_i)

        x = self.to_latent(x)
        x = self.to_pixel(x)
        x = self.decoder(x)
        x = F.sigmoid(x)

        return x
    
    def load(self, path="ckpt.pth"):
        tm = torch.load(path,map_location="cpu")
        self.load_state_dict(tm.state_dict(), strict=False)

    def record(self, t):
        if self.collecting:
            self.gram_feats.append(t)

    def gram_feature_list(self,x):
        self.collecting = True
        self.gram_feats = []
        self.forward(x)
        self.collecting = False
        temp = self.gram_feats
        self.gram_feats = []
        return temp

    def get_min_max(self, data, power):
        mins = []
        maxs = []

        for i in range(0, len(data), 64):
            batch = data[i:i+64].cuda()
            feat_list = self.gram_feature_list(batch)

            for L, feat_L in enumerate(feat_list):
                if L==len(mins):
                    mins.append([None]*len(power))
                    maxs.append([None]*len(power))

                for p,P in enumerate(power):
                    g_p = G_p(feat_L, P)

                    current_min = g_p.min(dim=0, keepdim=True)[0]
                    current_max = g_p.max(dim=0, keepdim=True)[0]

                    if mins[L][p] is None:
                        mins[L][p] = current_min
                        maxs[L][p] = current_max
                    else:
                        mins[L][p] = torch.min(current_min,mins[L][p])
                        maxs[L][p] = torch.max(current_max,maxs[L][p])
        
        return mins,maxs

    def get_deviations(self,data,power,mins,maxs):
        deviations = []
        
        for i in range(0, len(data), 64):
            batch = data[i:i+64].cuda()
            feat_list = self.gram_feature_list(batch)
            batch_deviations = []
            for L,feat_L in enumerate(feat_list):
                dev = 0
                for p,P in enumerate(power):
                    g_p = G_p(feat_L,P)
                    
                    dev +=  (F.relu(mins[L][p]-g_p)/torch.abs(mins[L][p]+10**-6)).sum(dim=1,keepdim=True)
                    dev +=  (F.relu(g_p-maxs[L][p])/torch.abs(maxs[L][p]+10**-6)).sum(dim=1,keepdim=True)
                batch_deviations.append(dev.cpu().detach().numpy())
            batch_deviations = np.concatenate(batch_deviations,axis=1)
            deviations.append(batch_deviations)
        deviations = np.concatenate(deviations,axis=0)

        return deviations