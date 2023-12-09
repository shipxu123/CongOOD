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
from net.ViG import *
from net.SwinHG import SwinUPer
from utils.metrics import call_metrics
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
from dataViG import ViGSet
from tqdm import tqdm

parser = argparse.ArgumentParser(description='ViG TRAINING')
parser.add_argument('--log', '-l', type=str, default="ViG.log")
parser.add_argument('--batch_size', '-b', type=int, default=4)
parser.add_argument('--resume', '-r', type=bool, default=False)
parser.add_argument('--exp', '-e', type=int, default=0)

EPOCH = 50
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

def save_ckpt(epoch, model, optimizer, scheduler, loss, ssim, nrms, save_path):
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss': loss,
        'ssim': ssim,
        'nrms': nrms
    }
    torch.save(checkpoint, save_path)

def train_vig():
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    exp = args.exp
    if exp == 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = SwinUPer(chIn=5, chOut=2, chMid=512, img_size=256, patch_size=4, embed_dim=96, 
                 window_size=7, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 depths=[2, 2, 18, 2], num_heads=[2, 4, 8, 16], out_indices=(0, 1, 2, 3))
    model.to(device)
    print(f"[INFO] USE {torch.cuda.device_count()} GPUs")
    #num_train = 392
    #num_test = 395
    #test = ViGSet("pending/testViG.pkl")
    train = ViGSet("trainData1.pkl")
    #n_train = len(train)
    gen1 = torch.Generator()
    gen1.manual_seed(0)
    
    
    #train = ViGSet("testData2.pkl")
    #n_test = len(test)
    #gen2 = torch.Generator()
    #gen2.manual_seed(0)
    #test, _ = torch.utils.data.random_split(test, [num_test, n_test-num_test], gen2)
    train, test = torch.utils.data.random_split(train, [0.7, 0.3], gen1)
    if exp == 0:
        train_loader = GraphDataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        test_loader = GraphDataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        with open("exp_B.pkl", "rb") as fin:
            test_files = pickle.load(fin)
            fin.close()
    else:
        train_loader = GraphDataLoader(test, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        test_loader = GraphDataLoader(train, batch_size=1, shuffle=False)
        with open("exp_A.pkl", "rb") as fin:
            test_files = pickle.load(fin)
            fin.close()

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCH//4, gamma=0.5)
    start_epoch = -1
    best_loss = 1e8
    best_ssim = 1e-4
    best_nrms = 1e4
    if args.resume:
        ckpt = torch.load("vig_ckpt.pth")
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        best_loss = ckpt['loss']
        best_ssim = ckpt['ssim']
        best_nrms = ckpt['nrms']
    logger_path = "./new_log1/"
    if not os.path.exists(logger_path):
          os.makedirs(logger_path)
          print("-----New Folder -----")
          print("----- " + logger_path + " -----")
          print("----- OK -----")
    save_log_path = logger_path + args.log
    logger = get_logger(save_log_path)
    logger.info('GOOOOO!')
    train_loss_recorder = []
    test_loss_recoder = []
    for epoch in range(start_epoch+1, EPOCH):
        train_loss = []
        test_loss = []
        ssim_res = []
        nrms_res = []
        print(f"-------- EPOCH {epoch+1} --------")
        model.train()
        #adjust_lr_with_warmup(optimizer, epoch+1, 4, 1024)
        #if epoch < 4:
        #    adjust_lr_with_warmup(optimizer, epoch+1, 8, 1024)
        #else:
        #    scheduler.step()
        #loss = 0

        optimizer.zero_grad()
        for i, (batch_x, batch_y) in enumerate(tqdm(train_loader)):
            #optimizer.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.float)
            batch_y = batch_y.to(device)
            congPred = model(batch_x)
            loss = loss_fn(batch_y, congPred)
            train_loss.append(loss.cpu().data.numpy())
            loss = loss/4
            loss.backward()
            if (i+1)%4 ==0:
                optimizer.step()
                optimizer.zero_grad()
    
            
        logger.info('[TRAIN] Epoch:{}/{}\t loss={:.8f}\t'.format(epoch+1, EPOCH, np.average(train_loss)))
        train_loss_recorder.extend(train_loss)
        optimizer.step()
        scheduler.step()
        model.eval()
        test_recorder_ssim = {}
        test_recorder_nrms = {}
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(tqdm(test_loader)):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                congPred = model(batch_x)
                #congPred = torch.zeros_like(batch_y, device=device)
                loss = loss_fn(batch_y, congPred)
                batch_y = batch_y.cpu().data.numpy()
                congPred = congPred.cpu().data.numpy()
                for b in range(0, congPred.shape[0]):
                    ssim, nrms = call_metrics(batch_y[b], congPred[b])
                    if ssim is None or nrms is None:
                        continue
                    ssim_res.append(ssim)
                    nrms_res.append(nrms)
                test_loss.append(loss.cpu().data.numpy())
                file_name = test_files[i]
                design_name, _ =  file_name.split("__")
                if design_name in test_recorder_ssim.keys():
                    test_recorder_ssim[design_name].append(ssim)
                    test_recorder_nrms[design_name].append(nrms)
                else:
                    test_recorder_ssim[design_name] = [ssim]
                    test_recorder_nrms[design_name] = [nrms]
        if np.mean(ssim_res)/np.mean(nrms_res) > best_ssim/best_nrms:
            best_loss = np.average(test_loss)
            best_ssim = np.average(ssim_res)
            best_nrms = np.average(nrms_res)
            save_ckpt(epoch, model, optimizer, scheduler, best_loss, best_ssim, best_nrms,logger_path + args.log[:-4] + ".pth")
        logger.info('[TEST]  Epoch:{}/{}\t loss={:.8f}\t SSIM={:.6f}\t NRMS={:.6f}'.format(epoch+1, EPOCH, np.average(test_loss),np.average(ssim_res),np.average(nrms_res)))
        test_loss_recoder.append(np.average(test_loss))
        #for key in test_recorder_ssim.keys():
        #    logger.info('[TEST]  Design={}\t: SSIM={:.6f}\t NRMS={:.6f}'.format(key, np.average(test_recorder_ssim[key]),np.average(test_recorder_nrms[key])))
        save_ckpt(epoch, model, optimizer, scheduler, best_loss, best_ssim, best_nrms,logger_path + args.log[:-4] + "_new.pth")
        #logger.info('[REC] best loss={:.8f}\t SSIM={:.6f}\t NRMS={:.6f}'.format(best_loss, best_ssim, best_nrms))
    with open(logger_path + args.log[:-4] + ".pkl", "wb") as fout:
            pickle.dump(train_loss_recorder, fout)
            fout.close()
    logger.info('ViG'+' END')
    

if __name__ == '__main__':
    seed = 0
    train_batch = 8000
    test_batch = 8000
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    num_gpus = 1
    
    # test = ViGSet("pending/testViG.pkl")
    # n_test = len(test)
    # gen2 = torch.Generator()
    # gen2.manual_seed(seed)
    # test, _ = torch.utils.data.random_split(test, [test_batch, n_test-test_batch], gen2)
    train_vig()