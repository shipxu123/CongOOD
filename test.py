import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import pickle
import numpy as np
import gzip
import glob
from tqdm import tqdm
from dataViG import ViGSet
from net.SwinHG import SwinUPer
import matplotlib.pyplot as plt

import dgl
import dgl.data
from dgl.dataloading import GraphDataLoader
# with open("exp_A.pkl", "rb") as fin:
#     filelist = pickle.load(fin)
#     fin.close()
# print(filelist)
# exit(1)
checkpoint = torch.load("new_log1/vig_modelr2_new.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinUPer(chIn=5, chOut=2, chMid=512, img_size=256, patch_size=4, embed_dim=96, 
                 window_size=7, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 depths=[2, 2, 18, 2], num_heads=[2, 4, 8, 16], out_indices=(0, 1, 2, 3))
model.load_state_dict(checkpoint['model'])
model.to(device)
dataset = ViGSet("trainData1.pkl")
model.eval()
# train_loader = GraphDataLoader(dataset, batch_size=2, shuffle=True)
# with open("draw.pkl", "rb") as fin:
#     data = pickle.load(fin)
#     fin.close()
# x, congY = data

#pred = model(batch_graph)
#print(pred.shape)
for i, batch in enumerate(tqdm(dataset)):
    if i > 5:
        break
    if i != 3:
        continue
    x, congY = batch
    x = x.to(device)
    pred = model(x)
    draw_congH = congY[0].cpu()



    draw_congV = congY[1].cpu()


    congH = (pred[0][0]).cpu().detach().numpy()
    
    congV = (pred[0][1]).cpu().detach().numpy()

    with open("laynet/pred_congH2.pkl", "wb") as fout:
        pickle.dump(congH, fout)
        fout.close()

    with open("laynet/gt_congH2.pkl", "wb") as fout:
        pickle.dump(draw_congH, fout)
        fout.close()

    with open("laynet/pred_congV2.pkl", "wb") as fout:
        pickle.dump(congV, fout)
        fout.close()

    with open("laynet/gt_congV2.pkl", "wb") as fout:
        pickle.dump(draw_congV, fout)
        fout.close()

    plt.subplot(221)
    plt.title("GT-congH")
    plt.axis('off')
    plt.imshow(draw_congH, cmap='YlGnBu_r', interpolation='nearest')

    plt.subplot(222)
    plt.title("GT-congV")
    plt.axis('off')
    plt.imshow(draw_congV, cmap='YlGnBu_r', interpolation='nearest')

    plt.subplot(223)
    plt.title("congH")
    plt.axis('off')
    plt.imshow(congH, cmap='YlGnBu_r', interpolation='nearest')

    plt.subplot(224)
    plt.title("congV")
    plt.axis('off')
    plt.imshow(congV, cmap='YlGnBu_r', interpolation='nearest')
    
    plt.savefig("laynet/"+str(i)+'.png')
    plt.close()


# from dataViG import *
# data = imageSet("trainViG.pkl")
# image, label = data[1]
# macro = image[-1]
# draw = np.zeros((macro.shape[0],macro.shape[1],3))
# draw[:,:,2] = macro.cpu()
# plt.imshow(draw)
# plt.savefig("macro.png")

