import pickle
import numpy as np
import os
os.environ['DGLBACKEND'] = 'pytorch'

import itertools
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as tmp
from einops.layers.torch import Rearrange

from utils.log import get_logger 
import dgl
import dgl.data
import dgl.sparse as dglsp
from dgl.dataloading import GraphDataLoader
from torchvision import transforms
import random
import argparse
import glob
import multiprocessing as mp
import time
import gzip

from tqdm import tqdm

from data.dataViG import ViGSet
from net.ViG import *
from net.SwinHG import SwinUPer
from utils.metrics import call_metrics

parser = argparse.ArgumentParser(description='ViG TRAINING')
parser.add_argument('--log', '-l', type=str, default="ViG.log")
parser.add_argument('--batch_size', '-b', type=int, default=8)
parser.add_argument('--resume', '-r', type=bool, default=False)

EPOCH = 100
TRAIN_BATCH = 180
TEST_BATCH = 20
NUM_GRID = 32
PATCH_SIZE = 8

def init_process_group(world_size, rank):
    dist.init_process_group(
        backend="nccl",  # change to 'nccl' for multiple GPUs
        init_method="tcp://127.0.0.1:12345",
        world_size=world_size,
        rank=rank,
    )

def load_feature(file_path):
    (path, filename) = os.path.split(file_path)
    with open(file_path, "rb") as fin:
        try:
            data = pickle.load(fin)
        except:
            fin.close()
            fin = gzip.GzipFile(file_path, "rb")
            data = pickle.load(fin)
        fin.close()
        v_n = data['v_n']
        w_nn = data['w_nn']
        e_cc = data['e_cc']
        e_cc = e_cc.tolist()
        e_cn = data['e_cn']
        num_gnet = e_cn[1][-1] + 1
        e_cn = e_cn.tolist()
        e_nn = data['e_nn']
        e_nn = e_nn.tolist()
        hg = dgl.heterograph({
                ('Gcell', 'cc', 'Gcell'): (torch.tensor(e_cc[0]), torch.tensor(e_cc[1])),
                ('Gcell', 'cn', 'Gnet'): (torch.tensor(e_cn[0]), torch.tensor(e_cn[1])),
                ('Gnet', 'nc', 'Gcell'): (torch.tensor(e_cn[1]), torch.tensor(e_cn[0])),
                ('Gnet', 'nn', 'Gnet'): (torch.tensor(e_nn[0]), torch.tensor(e_nn[1]))
                },
                {'Gcell': NUM_GRID*NUM_GRID, 'Gnet': num_gnet}
            )
        hg = dgl.add_self_loop(hg, etype='cc')
    with open("./new_collected/" + filename, "rb") as f:
        try:
            data = pickle.load(f)
        except:
            f.close()
            f = gzip.GzipFile("./new_collected/" + filename, "rb")
            data = pickle.load(f)
        v_c = np.concatenate((data['density'][None,:,:],data['rudy'][None,:,:],data['macro'][None,:,:],data['macro_h'][None,:,:],data['macro_v'][None,:,:]))
        v_c = torch.tensor(v_c, dtype=torch.float)
        transforms_size = transforms.Resize(size=(256,256), antialias=True)
        v_c = transforms_size(v_c)
        rearrange_vc = Rearrange('c (h p1) (w p2) -> (h w) (p1 p2 c)', p1 = PATCH_SIZE, p2 = PATCH_SIZE)
        v_c = rearrange_vc(v_c)
        hg.ndata['h'] = {'Gcell':v_c, 'Gnet':torch.tensor(v_n, dtype=torch.float)}
        hg.edata['h'] = {'nn':torch.tensor(w_nn)}
        congV = data['congV']
        congH = data['congH']
        congY = np.concatenate((congV[None,:,:],congH[None,:,:])) 
        congY = torch.tensor(congY)
        congY = transforms_size(congY)   
    return (hg, congY)

def load_data(train_ratio=0.9):
    filenames = glob.glob(f"data/ViG/*1.pkl")
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
        print(f"\r loaded {idx+threads}/{len(filenames)}, batch time={time.time()-begin}s",end="")
    print("Finish")
    data = list(data.values())
    random.shuffle(data)
    train_num = int(len(filenames)*train_ratio)
    return data[:train_num], data[train_num:]

def adjust_lr_with_warmup(optimizer, step, warm_up_step, dim):
    #lr = dim**(-0.5) * min(step**(-0.5), step*warm_up_step**(-1.5))/100
    lr = min(step**(-0.5), step*warm_up_step**(-1.5))*4e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_ckpt(epoch, model, optimizer, scheduler, loss, ssim, nrms):
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss': loss,
        'ssim': ssim,
        'nrms': nrms
    }
    torch.save(checkpoint, "vig_ckpt.pth")


import calculate_utils as callog
def detect(test_deviations,ood_deviations, verbose=True, normalize=True):
    average_results = {}
    for i in range(1,11):
        random.seed(i)
        
        validation_indices = random.sample(range(len(test_deviations)),int(0.1*len(test_deviations)))
        test_indices = sorted(list(set(range(len(test_deviations)))-set(validation_indices)))

        validation = test_deviations[validation_indices]
        test_deviations = test_deviations[test_indices]

        t95 = validation.mean(axis=0)+10**-7
        if not normalize:
            t95 = np.ones_like(t95)
        test_deviations = (test_deviations/t95[np.newaxis,:]).sum(axis=1)
        ood_deviations = (ood_deviations/t95[np.newaxis,:]).sum(axis=1)
        
        results = callog.compute_metric(-test_deviations,-ood_deviations)
        for m in results:
            average_results[m] = average_results.get(m,0)+results[m]
    
    for m in average_results:
        average_results[m] /= i
    if verbose:
        callog.print_results(average_results)
    return average_results


def cpu(ob):
    for i in range(len(ob)):
        for j in range(len(ob[i])):
            ob[i][j] = ob[i][j].cpu()
    return ob
    
def cuda(ob):
    for i in range(len(ob)):
        for j in range(len(ob[i])):
            ob[i][j] = ob[i][j].cuda()
    return ob

import calculate_utils as callog

def detect(test_deviations, ood_deviations, verbose=True, normalize=True):
    average_results = {}
    for i in range(1,11):
        random.seed(i)
        
        validation_indices = random.sample(range(len(test_deviations)),int(0.1*len(test_deviations)))
        test_indices = sorted(list(set(range(len(test_deviations)))-set(validation_indices)))

        validation = test_deviations[validation_indices]
        test_deviations = test_deviations[test_indices]

        t95 = validation.mean(axis=0)+10**-7
        if not normalize:
            t95 = np.ones_like(t95)
        test_deviations = (test_deviations/t95[np.newaxis,:]).sum(axis=1)
        ood_deviations = (ood_deviations/t95[np.newaxis,:]).sum(axis=1)
        
        results = callog.compute_metric(-test_deviations,-ood_deviations)
        for m in results:
            average_results[m] = average_results.get(m,0)+results[m]
    
    for m in average_results:
        average_results[m] /= i
    if verbose:
        callog.print_results(average_results)
    return average_results


def cpu(ob):
    for i in range(len(ob)):
        for j in range(len(ob[i])):
            ob[i][j] = ob[i][j].cpu()
    return ob
    
def cuda(ob):
    for i in range(len(ob)):
        for j in range(len(ob[i])):
            ob[i][j] = ob[i][j].cuda()
    return ob

class Detector:
    def __init__(self, model):
        self.test_deviations = None
        self.mins = {}
        self.maxs = {}
        self.model = model
        self.classes = range(10)
    
    def compute_minmaxs(self, train, POWERS=[10]):
        mins, maxs = self.model.get_min_max(train, power=POWERS)
        self.mins[0] = cpu(mins)
        self.maxs[0] = cpu(maxs)
        torch.cuda.empty_cache()
    
    def compute_test_deviations(self, test_preds, POWERS=[10]):
        test_deviations = None

        mins = cuda(self.mins[0])
        maxs = cuda(self.maxs[0])
        test_deviations = self.model.get_deviations(test_preds, power=POWERS, mins=mins, maxs=maxs) / test_preds[:, np.newaxis]
        cpu(mins)
        cpu(maxs)
        torch.cuda.empty_cache()
        self.test_deviations = test_deviations

    def compute_ood_deviations(self, ood_loader, POWERS=[10]):
        ood_preds = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(tqdm(ood_loader)):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                congPred = self.model(batch_x)
                preds = (congPred.cpu().detach().numpy())
                ood_preds.extend(preds)
        print("Done")

        mins = cuda(self.mins[0])
        maxs = cuda(self.maxs[0])
        ood_deviations = self.model.get_deviations(ood_preds, power=POWERS, mins=mins,maxs=maxs) / ood_preds[:,np.newaxis]
        cpu(self.mins[0])
        cpu(self.maxs[0])

        torch.cuda.empty_cache()

        average_results = detect(self.test_deviations, ood_deviations)
        return average_results, self.test_deviations, ood_deviations


def G_p(ob, p):
    temp = ob.detach()
    
    temp = temp**p
    temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
    temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1)))).sum(dim=2) 
    temp = (temp.sign()*torch.abs(temp)**(1/p)).reshape(temp.shape[0],-1)
    
    return temp

def detect_vig(train_dataset_dir, test_dataset_dir, ood_dataset_dir):
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinUPer(chIn=5, chOut=2, chMid=512, img_size=256, patch_size=4, embed_dim=128, 
                 window_size=7, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 depths=[2, 8, 18, 4], num_heads=[4, 8, 16, 32], out_indices=(0, 1, 2, 3))
    model.to(device)
    print(f"[INFO] USE {torch.cuda.device_count()} GPUs")

    num_train = 800
    num_test = 800
    num_ood = 800

    train = ViGSet(train_dataset_dir)

    n_train = len(train)
    gen1 = torch.Generator()
    gen1.manual_seed(0)
    train, _ = torch.utils.data.random_split(train, [num_train, n_train - num_train], gen1)

    test = ViGSet(test_dataset_dir)

    n_test = len(test)
    gen2 = torch.Generator()
    gen2.manual_seed(0)
    test, _ = torch.utils.data.random_split(test, [num_test, n_test-num_test], gen2)

    ood = ViGSet(ood_dataset_dir)
    n_ood= len(ood)
    gen3 = torch.Generator()
    gen3.manual_seed(0)
    ood, _ = torch.utils.data.random_split(n_ood, [num_ood, n_test-num_ood], 3)

    train_loader = GraphDataLoader(test, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader  = GraphDataLoader(train, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    ood_loader   = GraphDataLoader(ood, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    if args.resume:
        ckpt = torch.load("vig_ckpt.pth")
        model.load_state_dict(ckpt['model'])

    train_preds = []
    for i, (batch_x, batch_y) in enumerate(tqdm(train_loader)):
        batch_x = batch_x.to(device)
        batch_y = torch.tensor(batch_y, dtype=torch.float).to(device)

        congPred = model(batch_x)
        preds = (congPred.cpu().detach().numpy())
        train_preds.extend(preds)

    test_preds = []
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(tqdm(test_loader)):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            congPred = model(batch_x)
            preds = (congPred.cpu().detach().numpy())
            test_preds.extend(preds)

    print("Done")

    detector = Detector(model)
    detector.compute_minmaxs(train, POWERS=range(1,11))
    detector.compute_test_deviations(POWERS=range(1,11))

    print(f"{ood_dataset_dir}")
    ood_results = detector.compute_ood_deviations(ood, POWERS=range(1,11))
    print(ood_results)


if __name__ == '__main__':
    seed = 0
    train_batch = 8000
    test_batch = 8000
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    num_gpus = 1

    detect_vig("mgc_matrix_mult_1.pkl", "mgc_matrix_mult_2.pkl","./mm1_to_mm2_log/")