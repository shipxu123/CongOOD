import pickle
import numpy as np
import os
import torch
from tqdm import tqdm
def collect_data(mode='uniform'):
    #mode = 'uniform' or 'min' 
    # add macro space margin
    path = "./collected/collected/"
    for file in tqdm(os.listdir(path)):
        if os.path.exists("./new_collected/" + file):
             continue
        with open(path+file, "rb") as fin:
            data = pickle.load(fin)
            macro = data['macro']
            graph = data['graph']
            nodes = graph['nodes']
            layout = data['range']
            macro_nodes = list(filter(lambda x: x['macro'] == True, nodes))
            macro_h = np.zeros(macro.shape)
            macro_v = np.zeros(macro.shape)
            for x in range(macro.shape[0]):
                for y in range(macro.shape[1]):
                    if macro[x, y] == 1:
                         macro_h[x][y] = 0
                         macro_v[x][y] = 0
                         continue
                    left_index = np.where(macro[x,:y] == 1)
                    if left_index[0].shape[0] == 0:
                         left = 0
                    else:
                         left = left_index[0][-1]
                    right_index = np.where(macro[x,y+1:] == 1)
                    if right_index[0].shape[0] == 0:
                         right = macro.shape[1]
                    else:
                         right = right_index[0][0] + y + 1
                    down_index = np.where(macro[:x,y] == 1)
                    if down_index[0].shape[0] == 0:
                         down = 0
                    else:
                         down = down_index[0][-1]
                    top_index = np.where(macro[x+1:,y] == 1)
                    if top_index[0].shape[0] == 0:
                         top = macro.shape[0]
                    else:
                         top = top_index[0][0] + x + 1

            if mode == 'uniform':
                macro_h[x][y] = 1 - (right - left)/macro.shape[1] if right != macro.shape[1] or left != 0 else 0
                macro_v[x][y] = 1 - (top - down)/macro.shape[0] if top != macro.shape[0] or down != 0 else 0
            else:
                macro_h[x][y] = 1 - 2*min((right - y , y - left))/macro.shape[1] if right != macro.shape[1] or left != 0 else 0
                macro_v[x][y] = 1 - 2*min((top - x, x - down))/macro.shape[0] if top != macro.shape[0] or down != 0 else 0

            for node in macro_nodes:
                x1 = int((node['posX'] - layout[0])/(layout[2] - layout[0])*macro.shape[0])
                x2 = int((node['posX'] + node['sizeX'] - layout[0])/(layout[2] - layout[0])*macro.shape[0]) + 1
                y1 = int((node['posY'] - layout[1])/(layout[3] - layout[1])*macro.shape[1])
                y2 = int((node['posY'] + node['sizeY'] - layout[1])/(layout[3] - layout[1])*macro.shape[1]) + 1
                x1 = macro.shape[0] - 1 if x1 > macro.shape[0] - 1 else x1
                x2 = macro.shape[0] - 1 if x2 > macro.shape[0] - 1 else x2
                y1 = macro.shape[1] - 1 if y1 > macro.shape[1] - 1 else y1
                y2 = macro.shape[1] - 1 if y2 > macro.shape[1] - 1 else y2
                for i in range(x1, x2 + 1):
                     for j in range(0, 3):
                          macro_h[i][y1+j] = 1
                          macro_v[i][y1+j] = 1
                          macro_h[i][y2-j] = 1
                          macro_v[i][y2-j] = 1                         
                for j in range(y1, y2 + 1):
                     for i in range(0, 3):
                          macro_h[x1+i][j] = 1
                          macro_v[x1+i][j] = 1
                          macro_h[x2-i][j] = 1
                          macro_v[x2-i][j] = 1
        data['macro_h'] = macro_h
        data['macro_v'] = macro_v
        save_dir = './new_collected/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print("-----New Folder -----")
            print("----- " + save_dir + " -----")
            print("----- OK -----")
        with open(save_dir + file, "wb") as fout:
            pickle.dump(data, fout)



if __name__ == "__main__":
    collect_data()