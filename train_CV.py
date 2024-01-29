import pickle
import numpy as np
import os
import torch

import torch.utils.data as Data
from net.ViT import *
from net.UNet import *
from net.SwinViT import *
from utils.metrics import call_metrics
from utils.log import get_logger
from pending.data import * 
from torchvision import transforms
import random
import argparse
import glob
import multiprocessing as mp
import gzip

parser = argparse.ArgumentParser(description='CV TRAINING')
parser.add_argument('--log', '-l', type=str, default="CV.log")
parser.add_argument('--batch_size', '-b', type=int, default=32)
parser.add_argument('--model', '-m', type=str, default="ViT")

EPOCH = 200
TRAIN_BATCH = 1800
TEST_BATCH = 200

def load_feature(filename):
        fin = gzip.GzipFile(filename, "rb")
        data = pickle.load(fin)
        input_shape = data['density'].shape[0]
        v_c = np.concatenate((data['density'][None,:,:],data['rudy'][None,:,:],data['macro_h'][None,:,:],data['macro_v'][None,:,:]))
        #v_c = np.concatenate((data['density'][None,:,:],data['rudy'][None,:,:],data['macro'][None,:,:]))
        v_c = torch.tensor(v_c)
        transforms_size = transforms.Resize(size=(256,256), antialias=True)
        v_c = transforms_size(v_c)
        congY = np.concatenate((data['congH'][None,:,:], data['congV'][None,:,:]))
        congY = torch.tensor(congY)
        congY = transforms_size(congY)
        datum = np.concatenate((v_c, congY))
        fin.close()
        return datum

def load_data(train_ratio=0.9):
    filenames = glob.glob(f"new_collected/*1.pkl")
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
        print(f"\r loaded {idx+threads}/{len(filenames)}, batch time={time.time()-begin:.1f}s",end="")
    print("Finish")
    data = list(data.values())
    random.shuffle(data)
    data = np.array(data)
    data = torch.tensor(data)
    train_num = int(len(filenames)*train_ratio)
    return data[:train_num], data[train_num:]

def adjust_lr_with_warmup(optimizer, step, warm_up_step, dim):
    lr = dim**(-0.5) * min(step**(-0.5), step*warm_up_step**(-1.5))/100
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_cv():
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "ViT":
        model = ViT2(
         image_size=256,
         patch_size=16,
         num_classes=2,
         dim=1024,
         depth = 12,
         heads=16,
         mlp_dim=3072,
         channels=5,
         dropout=0.1,
         emb_dropout=0.1
        )
    elif args.model == "UNet":
        model = UNet(3, 2)
    elif args.model == "Swin":
        model = SwinUPer(chIn=4, chOut=2, chMid=512, img_size=256, patch_size=4, embed_dim=128, 
                         window_size=7, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                         depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], out_indices=(0, 1, 2, 3))
    else:
        print(args.model + " doesn't exists //// Please Use [ViT] or [UNet]")
        return 0
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0,1])
    model.to(device)
    print(f"[INFO] USE {torch.cuda.device_count()} GPUs")
    train, test = load_data()
    trainset = Data.TensorDataset(train[:,:-2], train[:,-2:])
    testset = Data.TensorDataset(test[:,:-2], test[:,-2:])
    #train_dataset, test_dataset = imageset("./collected/collected", trainRatio=0.9, maxNum=None, crop=0.125, size=(256, 256), preload=True)
    #trainset, testset = imageload("pending/train.pkl", "pending/test.pkl", crop=0.125, size=(256, 256), preload=True)
    train_loader = Data.DataLoader(trainset,batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Data.DataLoader(testset, batch_size=BATCH_SIZE,shuffle=False)
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
        if args.model == "ViT":
            adjust_lr_with_warmup(optimizer, epoch+1, 8, 1024)
        model.train()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_x = torch.tensor(batch_x, dtype=torch.float).to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.float).to(device)
            congPred = model(batch_x)
            loss = loss_fn(batch_y, congPred)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().data.numpy())
        logger.info('[TRAIN] Epoch:{}/{}\t loss={:.8f}\t'.format(epoch+1, EPOCH, np.average(train_loss)))
        if args.model == "UNet":
            scheduler.step()
        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = torch.tensor(batch_x, dtype=torch.float).to(device)
                batch_y = batch_y.to(device)
                congPred = model(batch_x)
                loss = loss_fn(batch_y, congPred)
                batch_y = batch_y.cpu().data.numpy()
                congPred = congPred.cpu().data.numpy()
                for b in range(0, congPred.shape[0]):
                    ssim, nrms = call_metrics(batch_y[b], congPred[b])
                    ssim_res.append(ssim)
                    nrms_res.append(nrms)
                test_loss.append(loss.cpu().data.numpy())
        logger.info('[TEST]  Epoch:{}/{}\t loss={:.8f}\t SSIM={:.6f}\t NRMS={:.6f}'.format(epoch+1, EPOCH, np.average(test_loss),np.average(ssim_res),np.average(nrms_res)))      
    logger.info(args.model+' END')


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(0)
    train_cv()