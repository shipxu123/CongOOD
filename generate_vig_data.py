import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import gzip

def cal_overlap(bbox1, bbox2, v_n=None, mode='inter', num_former=0):
    '''
    mode = 'inter' / 'intra1' / 'intra2'
    '''
    A = bbox1.shape[0]
    B = bbox2.shape[0]
    xy_max = np.minimum(bbox1[:, np.newaxis, 2:].repeat(B, axis=1),
                        np.broadcast_to(bbox2[:,2:], (A, B, 2)))
    xy_min = np.maximum(bbox1[:, np.newaxis, :2].repeat(B, axis=1),
                        np.broadcast_to(bbox2[:,:2], (A, B, 2)))
    inter = np.clip(xy_max - xy_min + 1, a_min=0, a_max=np.inf)
    area = inter[:, :, 0] * inter[:, :, 1]
    if mode == 'intra1' or mode == 'intra2':
        v_n = np.diag(v_n)
        area = area - v_n
    idx = area.nonzero()
    area = area[idx]
    idx = np.array(idx)
    if mode == 'inter':
        idx[1] = idx[1] + A
    elif mode == 'intra2':
        idx = idx + num_former
    return idx, area

def gen_vig_data(NUM_GRID=256):
    print("Generate Our InterConnection Graph")
    n = len(os.listdir("./new_collected"))
    count = 0
    for file in os.listdir("./new_collected"):
        if os.path.exists("./data/ViG/" + file):
             count += 1
             continue
        file_path = "./new_collected/" + file
        #with open(file_path, "rb") as f:
        print("[Processing] " + file[:-4] + f"------[{count+1}/{n}]")
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        except:
            f = gzip.GzipFile(file_path, "rb")
            data = pickle.load(f)
        f.close()
        layout_range = data["range"]
        x0, y0, x1, y1 = layout_range
        BIN_SIZE_X = (x1 - x0) / NUM_GRID
        BIN_SIZE_Y = (y1 - y0) / NUM_GRID
        graph = data["graph"]
        nodes = graph["nodes"]
        nets = sorted(graph["nets"],key=len)
        num_nets = len(nets)
        edge_pair_sorted = []
        bbox_list = []
        for i in range(num_nets):
            print(f"[Connecting Graph]----------{i}/{num_nets}")
            print(f"This Net is connecting with {len(nets[i])} Cells!")
            bin_index = []
            max_x = -1e8
            min_x = 1e8
            max_y = -1e8
            min_y = 1e8
            for j in range(0, len(nets[i])):
                bin_index_x = int((nodes[nets[i][j]]["posX"] - x0) / BIN_SIZE_X)
                bin_index_y = int((nodes[nets[i][j]]["posY"] - y0) / BIN_SIZE_Y)
                bin_index_x = NUM_GRID - 1 if bin_index_x > NUM_GRID - 1 else bin_index_x
                bin_index_y = NUM_GRID - 1 if bin_index_y > NUM_GRID - 1 else bin_index_y
                bin_index_t = bin_index_y * NUM_GRID + bin_index_x
                bin_index_x_max = int((nodes[nets[i][j]]["posX"] + nodes[nets[i][j]]["sizeX"] - x0) / BIN_SIZE_X) + 1
                bin_index_y_max = int((nodes[nets[i][j]]["posY"] + nodes[nets[i][j]]["sizeY"]-  y0) / BIN_SIZE_Y) + 1
                bin_index_x_max = NUM_GRID - 1 if bin_index_x_max > NUM_GRID - 1 else bin_index_x_max
                bin_index_y_max = NUM_GRID - 1 if bin_index_y_max > NUM_GRID- 1 else bin_index_y_max
                max_x = bin_index_x_max if bin_index_x_max > max_x else max_x
                min_x = bin_index_x if bin_index_x < min_x else min_x
                max_y = bin_index_y_max if bin_index_y_max > max_y else max_y
                min_y = bin_index_y if bin_index_y < min_y else min_y
                if (max_x == NUM_GRID - 1 and max_y == NUM_GRID - 1 and min_x == 0 and min_y == 0):
                    break
                if bin_index_t not in bin_index:
                    bin_index.append(bin_index_t)
            print(f"This Net is crossing {len(bin_index)} Bins!")
            if(max_x - min_x + 1) * (max_y - min_y + 1) > NUM_GRID*NUM_GRID*0.016 or len(bin_index)/((max_x - min_x+1)*(max_y-min_y+1)) <0.25 or len(bin_index) < 2:
                continue
            else:
                bbox_list.append((min_x, min_y, max_x, max_y))
                bin_index_sorted = sorted(bin_index)
                edge_pair_temp =  list(itertools.combinations(bin_index_sorted, 2))
                edge_pair_temp =  [list(group) for val, group in itertools.groupby(edge_pair_temp, lambda x: abs(x[0]/NUM_GRID - x[1]/NUM_GRID) < 10 and abs(x[0]%NUM_GRID - x[1]%NUM_GRID) < 10) if val]
                if (len(edge_pair_temp)>0):
                    edge_pair_sorted = list(edge_pair_sorted + edge_pair_temp[0])
        edge_pair_sorted = list(set(edge_pair_sorted))
        edge_pair = np.array(edge_pair_sorted).T
        edge_pair_reversed = edge_pair[[1,0],:]
        edge_index = np.concatenate((edge_pair, edge_pair_reversed),axis=1)
        bbox_list = list(set(bbox_list))
        num_hyperedge = len(bbox_list)
        v_n = np.zeros((num_hyperedge, 3))
        source_index = [] # edge_index (GCell --> GNet)
        target_index = []
        edge_cn_index = [] 
        print("[Generate GNet Feature]")
        for i in range(num_hyperedge):
            print(f"[Process]----- {i+1}/{num_hyperedge} GNets")
            min_x, min_y, max_x, max_y = bbox_list[i]
            span_h = max_x - min_x + 1
            span_v = max_y - min_y + 1
            area = span_h * span_v
            v_n[i][0] = span_h
            v_n[i][1] = span_v
            v_n[i][2] = area
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    source_index.append(y * NUM_GRID + x)
                    target_index.append(i)
        edge_cn_index.append(source_index)
        edge_cn_index.append(target_index)
        edge_cn_index = np.array(edge_cn_index)
        overlapping_pairs = []
        overlapping_area = []
        bbox_list = np.array(bbox_list)
        n = bbox_list.shape[0]
        bbox1 = bbox_list[:n//2]
        v_n1 = v_n[:n//2,2]
        bbox2 = bbox_list[n//2:]
        v_n2 = v_n[n//2:,2] 
        idx1, area1 = cal_overlap(bbox1, bbox2, mode='inter')
        idx2, area2 = cal_overlap(bbox1, bbox1, v_n1, mode='intra1')
        idx3, area3 = cal_overlap(bbox2, bbox2, v_n2, mode='intra2', num_former=bbox1.shape[0])
        overlapping_pairs = np.concatenate((idx1, idx2, idx3), axis=-1)
        overlapping_area = np.concatenate((area1, area2, area3))
        save_path = './data/ViG/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print("-----New Folder -----")
            print("----- " + save_path + " -----")
            print("----- OK -----")
        data['e_cc'] = edge_index
        data['e_nn'] = overlapping_pairs
        data['w_nn'] = overlapping_area
        data['v_n'] = v_n
        data['e_cn'] = edge_cn_index
        fout = gzip.GzipFile(save_path+file, "wb")
        pickle.dump(data, fout)
        fout.close()
        count += 1

if __name__ == "__main__":
    gen_vig_data(32)