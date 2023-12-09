import pickle
import numpy as np
import os
import glob
import itertools
import torch
import torch.nn.functional as F
from net.LHNN import *
from dataViG import *
from utils.metrics import call_metrics
from utils.log import get_logger 
import dgl
import dgl.data
import dgl.sparse as dglsp
from dgl.dataloading import GraphDataLoader
from torchvision import transforms
import random
import argparse
from tqdm import tqdm
import time
import multiprocessing as mp

parser = argparse.ArgumentParser(description='LHNN TRAINING')
parser.add_argument('--log', '-l', type=str, default="LHNN.log")
parser.add_argument('--batch_size', '-b', type=int, default=32)

NUM_GRID = 256
EPOCH = 100
TRAIN_BATCH = 800
TEST_BATCH = 200

def generate_adj(hg, NUM_GRID=256):
    '''
    hg = dgl.heterograph({
         ('Gcell', 'cc', 'Gcell'): (torch.tensor(b[0]), torch.tensor(b[1])),
         ('Gcell', 'cn', 'Gnet'): (torch.tensor(a[0]), torch.tensor(a[1]))
    })
    '''
    batch_size = hg.batch_size
    a_cc = hg.edges(etype=('cc'))
    a_cc = np.array([a_cc[0].numpy(),a_cc[1].numpy()])
    a_cc = torch.from_numpy(a_cc)
    a_cc = dglsp.spmatrix(a_cc, shape=(batch_size*(NUM_GRID**2),batch_size*(NUM_GRID**2)))
    h_cn = hg.edges(etype=('cn'))
    h_cn = np.array([h_cn[0].numpy(),h_cn[1].numpy()])
    h_cn = torch.from_numpy(h_cn)
    try:
        num_hyperedge = h_cn[1][-1]+1
    except:
        num_hyperedge = 0
    h_cn = dglsp.spmatrix(h_cn,shape=(batch_size*(NUM_GRID**2),hg.in_degrees(etype=('cn')).shape[0]))
    degree =  hg.in_degrees(etype=('cc')) + hg.out_degrees(etype=('cc'))
    p = dglsp.diag(degree).inv()
    degree =  hg.in_degrees(etype=('cn'))
    b = dglsp.diag(degree).inv()
    A = dglsp.matmul(p, a_cc)
    G_cn = dglsp.matmul(b, h_cn.coalesce().T)
    G_nc = h_cn
    return A, G_cn, G_nc

def load_feature(filepath):
    (path, file) = os.path.split(filepath)
    lattice_graph = np.load("./data/LHNN/lattice_graph.npy")
    lattice_graph = lattice_graph.tolist()
    v_n = np.load(filepath+"/v_n.npy")
    hypergraph = np.load(filepath+"/hyperedge_index.npy")
    hypergraph = hypergraph.tolist()
    hg = dgl.heterograph({
        ('Gcell', 'cc', 'Gcell'): (torch.tensor(lattice_graph[0]), torch.tensor(lattice_graph[1])),
        ('Gcell', 'cn', 'Gnet'): (torch.tensor(hypergraph[0]), torch.tensor(hypergraph[1])),
    })
    with open("./new_collected/" + file + ".pkl", "rb") as f:
        data = pickle.load(f)
    v_c = np.concatenate((data['density'][None,:,:],data['rudy'][None,:,:],data['macro'][None,:,:]))
    v_c = torch.tensor(v_c, dtype=torch.float)
    transforms_size = transforms.Resize(size=(256,256),antialias=True)
    v_c = transforms_size(v_c)
    v_c = v_c.permute(1,2,0)
    v_c = v_c.view(-1,3)
    hg.ndata['h'] = {'Gcell': v_c, 'Gnet':torch.tensor(v_n, dtype=torch.float)}
    congY = np.concatenate((data['congH'][None,:,:], data['congV'][None,:,:]))
    congY = torch.tensor(congY)
    congY = transforms_size(congY)
    return (hg, congY)                       

def load_data(train_ratio=0.9):
    filenames = glob.glob(f"data/LHNN/*")
    filenames.remove("data/LHNN/lattice_graph.npy")
    data = {}
    threads = 64
    for idx in range(0, len(filenames), threads):
        begin = time.time()
        pool = mp.Pool(processes=threads)
        procs = []
        for jdx in range(min(threads, len(filenames)-idx)):
            proc = pool.apply_async(load_feature, (filenames[idx+jdx],))
            procs.append(proc)
        pool.close()
        pool.join()
        for jdx, proc in enumerate(procs):
            datum = proc.get()
            data[filenames[idx+jdx]] = datum
        del pool
        del procs
        print(f"\r loaded {idx+threads}/{len(filenames)}, batch time={time.time()-begin:.1f}s", end="")
    print("Finish")
    data = list(data.values())
    random.shuffle(data)
    train_num = int(len(filenames)*train_ratio)
    return data[:train_num], data[train_num:]


def train_LHNN():
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LHNN(3,3,32,2,256)
    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)
    model.to(device)
    print(f"[INFO] USE {torch.cuda.device_count()} GPUs")
    num_train = 5000
    num_test = 5000
    #test = ViGSet("pending/testViG.pkl")
    train = LHNNSet("pending/trainData.pkl")
    n_train = len(train)
    gen1 = torch.Generator()
    gen1.manual_seed(0)
    train, _ = torch.utils.data.random_split(train, [num_train, n_train - num_train], gen1)
    
    test = LHNNSet("pending/testData1.pkl")
    n_test = len(test)
    gen2 = torch.Generator()
    gen2.manual_seed(0)
    test, _ = torch.utils.data.random_split(test, [num_test, n_test-num_test], gen2)
    train_loader = GraphDataLoader(train,batch_size=BATCH_SIZE, shuffle=True)
    test_loader = GraphDataLoader(test,batch_size=BATCH_SIZE, shuffle=False)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCH//4, gamma=0.5)
    logger_path = "./log/"
    if not os.path.exists(logger_path):
          os.makedirs(logger_path)
          print("-----New Folder -----")
          print("----- " + logger_path + " -----")
          print("----- OK -----")
    save_log_path = logger_path + args.log
    logger = get_logger(save_log_path)
    logger.info('GOOOOO!')
    for epoch in range(0, EPOCH):
        train_loss = []
        test_loss = []
        ssim_res = []
        nrms_res = []
        print(f"-------- EPOCH {epoch+1} --------")
        model.train()
        for train_batch in tqdm(train_loader):
            optimizer.zero_grad()
            hg, congY = train_batch
            A, G_cn, G_nc = generate_adj(hg)
            hg, A, G_cn, G_nc, congY = hg.to(device), \
                                        A.to(device), \
                                        G_cn.to(device), \
                                        G_nc.to(device), \
                                        torch.tensor(congY, dtype=torch.float).to(device)
            congPred = model(hg, G_nc, G_cn, A)
            loss = loss_fn(congPred,congY)
            train_loss.append(loss.cpu().data.numpy())
            loss = loss
            loss.backward()
            optimizer.step()
        logger.info('[TRAIN] Epoch:{}/{}\t loss={:.8f}\t'.format(epoch+1, EPOCH, np.average(train_loss)))
        scheduler.step()
        model.eval()
        with torch.no_grad():
            for test_batch in tqdm(test_loader):
                hg, congY = test_batch
                A, G_cn, G_nc = generate_adj(hg)
                hg, A, G_cn, G_nc, congY = hg.to(device), \
                                        A.to(device), \
                                        G_cn.to(device), \
                                        G_nc.to(device), \
                                        torch.tensor(congY, dtype=torch.float).to(device)
                congPred = model(hg, G_nc, G_cn, A)
                loss = loss_fn(congPred,congY)
                congY = congY.cpu().data.numpy()
                congPred = congPred.cpu().data.numpy()
                for b in range(0, congPred.shape[0]):
                    ssim, nrms = call_metrics(congY[b], congPred[b])
                    if ssim is None or nrms is None:
                        continue
                    ssim_res.append(ssim)
                    nrms_res.append(nrms)
                test_loss.append(loss.cpu().data.numpy())
        logger.info('[TEST]  Epoch:{}/{}\t loss={:.8f}\t SSIM={:.6f}\t NRMS={:.6f}'.format(epoch+1, EPOCH, np.average(test_loss),np.average(ssim_res),np.average(nrms_res)))      
    logger.info('LHNN END')


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(0)
    train_LHNN()