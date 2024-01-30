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

def train_vig(train_dataset_dir, test_dataset_dir, log_dir):
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
    # train = ViGSet("trainData1.pkl")
    # train = ViGSet("fft_a.pkl")
    # train = ViGSet("mgc_des_perf_1.pkl")
    # train = ViGSet("mgc_des_perf_2.pkl")
    train = ViGSet(train_dataset_dir)

    n_train = len(train)
    gen1 = torch.Generator()
    gen1.manual_seed(0)
    train, _ = torch.utils.data.random_split(train, [num_train, n_train - num_train], gen1)

    # test = ViGSet("testData2.pkl")
    # test = ViGSet("fft_b.pkl")
    # test = ViGSet("mgc_des_perf_1.pkl")
    test = ViGSet(test_dataset_dir)

    n_test = len(test)
    gen2 = torch.Generator()
    gen2.manual_seed(0)
    test, _ = torch.utils.data.random_split(test, [num_test, n_test-num_test], gen2)
    train_loader = GraphDataLoader(test, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = GraphDataLoader(train, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCH//4, gamma=0.5)
    start_epoch = -1
    best_loss = 1e8
    best_ssim = 0
    best_nrms = 0

    if args.resume:
        ckpt = torch.load("vig_ckpt.pth")
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        best_loss = ckpt['loss']
        best_ssim = ckpt['ssim']
        best_nrms = ckpt['nrms']

    # logger_path = "./new_log/"
    logger_path = log_dir

    if not os.path.exists(logger_path):
          os.makedirs(logger_path)
          print("-----New Folder -----")
          print("----- " + logger_path + " -----")
          print("----- OK -----")
    save_log_path = logger_path + args.log
    logger = get_logger(save_log_path)
    logger.info('GOOOOO!')

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
            batch_y = torch.tensor(batch_y, dtype=torch.float).to(device)
            congPred = model(batch_x)
            loss = loss_fn(batch_y, congPred)
            train_loss.append(loss.cpu().data.numpy())
            loss = loss/4
            loss.backward()
            if (i+1)%4 ==0:
                optimizer.step()
                optimizer.zero_grad()

        logger.info('[TRAIN] Epoch:{}/{}\t loss={:.8f}\t'.format(epoch+1, EPOCH, np.average(train_loss)))
        optimizer.step()
        scheduler.step()
        model.eval()

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
        if np.average(test_loss) < best_loss:
            best_loss = np.average(test_loss)
            best_ssim = np.average(ssim_res)
            best_nrms = np.average(nrms_res)
        save_ckpt(epoch, model, optimizer, scheduler, best_loss, best_ssim, best_nrms)
        logger.info('[TEST]  Epoch:{}/{}\t loss={:.8f}\t SSIM={:.6f}\t NRMS={:.6f}'.format(epoch+1, EPOCH, np.average(test_loss),np.average(ssim_res),np.average(nrms_res)))
        logger.info('[REC] best loss={:.8f}\t SSIM={:.6f}\t NRMS={:.6f}'.format(best_loss, best_ssim, best_nrms))
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
    # train_vig()
    train_vig("mgc_matrix_mult_1.pkl", "mgc_matrix_mult_2.pkl","./mm1_to_mm2_log/")