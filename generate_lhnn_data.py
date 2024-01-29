import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import gzip

def lattice_graph(NUM_GRID=256):
     print("Generate Hypergraph of LHNN (GCell part)")
     source_node = []
     target_node = []
     edge_index = []
     for i in range(0, NUM_GRID):
          for j in range(0, NUM_GRID):
               if(j < NUM_GRID - 1):
                    source_node.append(i*NUM_GRID+j)
                    target_node.append(i*NUM_GRID+j+1)
               if(i < NUM_GRID - 1):
                    source_node.append(i*NUM_GRID+j)
                    target_node.append((i+1)*NUM_GRID+j)
     edge_index.append(source_node)
     edge_index.append(target_node)
     edge_index = np.array(edge_index)
     edge_index_reversed = edge_index[[1,0],:]
     edge_index = np.concatenate((edge_index,edge_index_reversed),axis=1)
     print(f"Lattice Grpah contains {edge_index.shape[1]} edges")
     save_path = "./data/LHNN/"
     if not os.path.exists(save_path):
          os.makedirs(save_path)
          print("-----New Folder -----")
          print("----- " + save_path + " -----")
          print("----- OK -----")
     np.save(save_path + "lattice_graphx1.npy", edge_index)

def hypergraph_lhnn(NUM_GRID=256):
    print("Generate Hypergraph of LHNN (GNet part)")
    print(os.listdir("./new_collected"))
    n = len(os.listdir("./new_collected"))
    count = 0
    for file in os.listdir("./new_collected"):
        if os.path.exists("./data/LHNN/" + file[:-4] + "/"):
             count += 1
             continue
        file_path = "./new_collected/" + file
        with open(file_path, "rb") as f:
            print("[Processing] " + file[:-4] + f"------[{count+1}/{n}]")
            try:
               data = pickle.load(f)
            except:
               f.close()
               f = gzip.GzipFile(file_path, "rb")
               data = pickle.load(f)
               f.close()
            layout_range = data["range"]
            x0, y0, x1, y1 = layout_range
            BIN_SIZE_X = (x1 - x0) / NUM_GRID
            BIN_SIZE_Y = (y1 - y0) / NUM_GRID
            print(layout_range)
            graph = data["graph"]
            print(graph.keys())
            nodes = graph["nodes"]
            nets = sorted(graph["nets"],key=len)
            num_nets = len(nets)
            hyperedge = []
            for i in range(num_nets):
                print(f"This Net is connecting with {len(nets[i])} Cells!")
                print("Calucating the bouning box")
                print(f"[Processing]----------{i}/{num_nets}")
                bin_index = []
                max_x = -1e8
                min_x = 1e8
                max_y = -1e8
                min_y = 1e8
                for j in range(0, len(nets[i])):
                     bin_index_x_min = int((nodes[nets[i][j]]["posX"] - x0) / BIN_SIZE_X)
                     bin_index_y_min = int((nodes[nets[i][j]]["posY"] - y0) / BIN_SIZE_Y)
                     bin_index_x_max = int((nodes[nets[i][j]]["posX"] + nodes[nets[i][j]]["sizeX"] - x0) / BIN_SIZE_X)
                     bin_index_y_max = int((nodes[nets[i][j]]["posY"] + nodes[nets[i][j]]["sizeY"]-  y0) / BIN_SIZE_Y)
                     bin_index_x_min = NUM_GRID - 1 if bin_index_x_min > NUM_GRID - 1 else bin_index_x_min
                     bin_index_y_min = NUM_GRID - 1 if bin_index_y_min > NUM_GRID - 1 else bin_index_y_min
                     bin_index_x_max = NUM_GRID - 1 if bin_index_x_max > NUM_GRID - 1 else bin_index_x_max
                     bin_index_y_max = NUM_GRID - 1 if bin_index_y_max > NUM_GRID- 1 else bin_index_y_max
                     max_x = bin_index_x_max if bin_index_x_max > max_x else max_x
                     min_x = bin_index_x_min if bin_index_x_min < min_x else min_x
                     max_y = bin_index_y_max if bin_index_y_max > max_y else max_y
                     min_y = bin_index_y_min if bin_index_y_min < min_y else min_y
                     if (max_x == NUM_GRID - 1 and max_y == NUM_GRID - 1 and min_x == 0 and min_y == 0):
                          break
                if(max_x - min_x + 1) * (max_y - min_y + 1) > NUM_GRID*NUM_GRID*0.25/100:
                     print("Remove Large Degree G-net")
                     continue
                hyperedge = hyperedge + [(min_x, max_x, min_y, max_y)]
            hyperedge = list(set(hyperedge))
            print(f"Total {len(hyperedge)} GNets")
            num_hyperedge = len(hyperedge)
            v_n = np.zeros((num_hyperedge, 3))
            source_index = [] # edge_index (GCell --> GNet)
            target_index = []
            edge_index = [] 
            print("[Generate GNet Feature]")
            for i in range(num_hyperedge):
                 print(f"[Process]----- {i+1}/{num_hyperedge} GNets")
                 min_x, max_x, min_y, max_y = hyperedge[i]
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
            edge_index.append(source_index)
            edge_index.append(target_index)
            edge_index = np.array(edge_index)
            save_path = './data/LHNN/' + file[:-4] + "/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                print("-----New Folder -----")
                print("----- " + save_path + " -----")
                print("----- OK -----")
            np.save(save_path + "hyperedge_index.npy", edge_index)
            np.save(save_path + "v_n.npy",v_n)
            count += 1

if __name__ == "__main__":
    lattice_graph(256)
    #hypergraph_lhnn(256)