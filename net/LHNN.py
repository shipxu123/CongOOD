import torch
from torch import nn
import dgl
import dgl.data
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import torch.nn.functional as F

class HeteroResidualBlock(nn.Module):
    def __init__(self, dim):
        super(HeteroResidualBlock, self).__init__()
        self.lin1 = dglnn.HeteroLinear({'Gcell':dim, 'Gnet':dim}, 2*dim)
        self.lin2 = dglnn.HeteroLinear({'Gcell':2*dim, 'Gnet':2*dim}, dim)

    def forward(self, h):
        h1 = self.lin1(h)
        h1 = {k:F.relu(v) for k, v in h1.items()}
        h2 = self.lin2(h1)
        h['Gcell'] = h['Gcell'] + h2['Gcell']
        h['Gnet'] = h['Gnet'] + h2['Gnet']
        h = {k:F.relu(v) for k, v in h.items()}
        return h

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.lin1 = nn.Linear(dim, 2*dim)
        self.lin2 = nn.Linear(2*dim, dim)

    def forward(self, h):
        h1 = self.lin2(F.relu(self.lin1(h)))
        return F.relu(h + h1)

class FeatureGen(nn.Module):
    def __init__(self, c_dim, n_dim, hidden_dim):
        super().__init__()
        self.embedding1 = dglnn.HeteroLinear({'Gcell': c_dim, 'Gnet':n_dim}, hidden_dim)
        self.embedding2 = dglnn.HeteroLinear({'Gcell': 2*hidden_dim, 'Gnet':hidden_dim}, hidden_dim)
        self.res = HeteroResidualBlock(hidden_dim)
    
    def forward(self, graph, g_nc):
        h = graph.ndata['h']
        h = self.embedding1(h)
        h = {k:F.relu(v) for k, v in h.items()}
        h = self.res(h)
        v_n = h['Gnet']
        v_c = h['Gcell']
        if g_nc.shape[-1] == v_n.shape[0]:
            v_c = torch.cat([v_c, g_nc @ v_n], dim=-1)
        else:
            v_c = torch.cat([v_c,torch.zeros((v_c.shape[0], v_n.shape[-1]), dtype='cuda:1')], dim=-1)
        h['Gcell'] = v_c
        h = self.embedding2(h)
        h = {k:F.relu(v) for k, v in h.items()}
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.update_all(
                message_func = fn.copy_u('h','m'),
                reduce_func = fn.sum('m', 'h_N')
            )
            h['Gcell'] = graph.ndata['h_N']['Gcell'] + h['Gcell']
            return h
        
class HyperMP_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin_n1 = nn.Linear(dim, dim)
        self.lin_c1 = nn.Linear(dim, dim)
        self.lin_cn1 = nn.Linear(dim, dim)
        self.lin_cn2 = nn.Linear(2*dim, dim)
        self.res1 = HeteroResidualBlock(dim)
        self.res_n = ResidualBlock(dim)
        self.res_c = ResidualBlock(dim)
        self.lin_nc1 = nn.Linear(dim, dim)
        self.lin_nc2 = nn.Linear(2*dim, dim)

    def forward(self, graph, h, h1, g_cn, g_nc):
        #G-cell to G-net
        h = self.res1(h)
        v_n = h['Gnet']
        v_c = h['Gcell']
        v_n1 = h1['Gnet']
        v_c1 = h1['Gcell']
        v_n1 = F.relu(self.lin_n1(v_n1))
        v_c1 = F.relu(self.lin_c1(v_c1))
        try:
            z = g_cn @ v_c
        except:
            z = torch.zeros((v_n1.shape[0], v_c.shape[-1]), device='cuda:1') 
        v_n = v_n + F.relu(self.lin_cn2(torch.cat([v_n1, F.relu(self.lin_cn1(z))], dim=-1)))
        #G-net to G-cell
        v_n = self.res_n(v_n)
        v_c = self.res_c(v_c)
        try:
            z = g_nc @ v_n
        except:
            z = torch.zeros((v_c1.shape[0], v_n.shape[-1]), device='cuda:1')
        v_c = v_c + F.relu(self.lin_nc2(torch.cat([v_c1, F.relu(self.lin_nc1(g_nc @ v_n))], dim=-1)))
        h['Gcell'] = v_c
        h['Gnet'] = v_n
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.update_all(
                message_func = fn.copy_u('h','m'),
                reduce_func = fn.sum('m', 'h_N')
            )
            h['Gcell'] = graph.ndata['h_N']['Gcell'] + h['Gcell']
            return h
        
class LatticeMP_Block(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.res1 = ResidualBlock(dim)
        self.lin1 = nn.Linear(dim, out_dim)
        self.lin2 = nn.Linear(dim, out_dim)

    def forward(self, graph, h, A):
        v_c = h['Gcell']
        v_c = self.res1(v_c)
        v_c = F.relu(self.lin1(v_c)) + F.relu(self.lin2(A @ v_c))
        h['Gcell'] = v_c
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.update_all(
                message_func = fn.copy_u('h','m'),
                reduce_func = fn.sum('m', 'h_N')
            )
            h['Gcell'] = graph.ndata['h_N']['Gcell'] + h['Gcell']
            return h['Gcell']
        
class LHNN(nn.Module):
    def __init__(self, c_dim, n_dim, hidden_dim, out_dim, img_size=256):
        super().__init__()
        self.featgen = FeatureGen(c_dim, n_dim, hidden_dim)
        self.hyper1 = HyperMP_Block(hidden_dim)
        self.hyper2 = HyperMP_Block(hidden_dim)
        self.lattice = LatticeMP_Block(hidden_dim, out_dim)
        self.out_dim = out_dim
        self.img_size = img_size
        self.sigmoid = nn.Sigmoid()

    def forward(self, graph, g_nc, g_cn, A):
        b = graph.batch_size
        sg = dgl.sampling.sample_neighbors(graph, {'Gcell': range(graph.num_nodes('Gcell')), 'Gnet': range(graph.num_nodes('Gnet'))}, 6)
        h1 = self.featgen(sg, g_nc)
        sg = dgl.sampling.sample_neighbors(graph, {'Gcell': range(graph.num_nodes('Gcell')), 'Gnet': range(graph.num_nodes('Gnet'))}, 3)
        h = self.hyper1(sg, h1, h1, g_cn, g_nc)
        sg = dgl.sampling.sample_neighbors(graph, {'Gcell': range(graph.num_nodes('Gcell')), 'Gnet': range(graph.num_nodes('Gnet'))}, 3)
        h = self.hyper2(sg, h, h1, g_cn, g_nc)
        sg = dgl.sampling.sample_neighbors(graph, {'Gcell': range(graph.num_nodes('Gcell')), 'Gnet': range(graph.num_nodes('Gnet'))}, 2)
        h = self.lattice(sg, h, A)
        h = self.sigmoid(h)
        h = h.permute(1,0)
        h = h.reshape(b, self.out_dim, self.img_size, self.img_size)
        return h



