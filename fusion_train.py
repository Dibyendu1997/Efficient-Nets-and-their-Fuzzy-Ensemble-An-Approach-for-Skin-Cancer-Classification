import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import GradualWarmupSchedulerV2
import apex
#from apex import amp
from dataset import get_df, get_transforms, MelanomaDataset
from fusion_model import Effnet_Melanoma
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='/raid/')
    parser.add_argument('--data-folder', type=int, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--init-lr', type=float, default=3e-5)
    parser.add_argument('--out-dim', type=int, default=8)
    parser.add_argument('--n-epochs', type=int, default=15)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./fusion_weights')
    parser.add_argument('--log-dir', type=str, default='./fusion_logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--fold', type=str, default='0,1,2,3,4')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    parser.add_argument('--latent-dim', type=int, default=1000)

    args, _ = parser.parse_known_args()
    return args


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    
def weightedsampler(target):
    
    class_sample_count = np.unique(target,return_counts = True)[1]
    #print(class_sample_count)
    
    weight = 1. / class_sample_count
    
    
    sample_weight = np.array([weight[t] for t in target])
    
    
    sample_weight =  torch.from_numpy(sample_weight)
    #print("sample weight",sample_weight.shape)
    
    return  sample_weight


def train_epoch(model, loader, optimizer):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:

        optimizer.zero_grad()
        
        if args.use_meta:   
            data, meta = data
            data, meta, target = data.to(device), meta.to(device), target.to(device)
            model_output_dict = model(data, meta)
            logits = model_output_dict["logits"]
            probs = logits.softmax(1)
            y_pred = probs.argmax(1)
            #print("pred",y_pred)
            #print("macrof1",f1_score(target.detach().cpu(), y_pred.detach().cpu(), average='micro'))
            score = f1_score(target.detach().cpu(), y_pred.detach().cpu(), average='micro')
            score = score + 0.000000001
            reward = 1/score
            reward = torch.tensor(reward)
            #print(logits.argmax(1)) 
        loss = criterion(logits, target)
        loss += reward
        if model_output_dict['aux_loss'] is not None:
        	#print("yes")
        	loss += 2*model_output_dict['aux_loss']  

        if not args.use_amp:
            loss.backward()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        if args.image_size in [896,576]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    return train_loss


def get_trans(img, I):

    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def val_epoch(model, loader, mel_idx, is_ext=None, n_test=1, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):
        
            
            if args.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    model_output_dict = model(get_trans(data, I), meta)
                    l = model_output_dict['logits']
                    logits += l
                    probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())
            loss = criterion(logits, target)
            if model_output_dict['aux_loss'] is not None:
            	loss += model_output_dict['aux_loss']
            	#print('yes')
            #if model_output_dict['aux_loss'] is not None:
                #loss += model_output_dict['aux_loss']   
            
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    print(PROBS.argmax(1), TARGETS)
    if get_output:
        return LOGITS, PROBS
    else:
        labels = [0, 1, 2, 3,4,5,6,7]
        TARGETS1 = label_binarize(TARGETS, classes=labels)
        print("Targets",TARGETS)
        
        print("Prediction",PROBS.argmax(1))
        PRED = label_binarize(PROBS.argmax(1), classes=labels)
        acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
        roc_auc = roc_auc_score(TARGETS1, PRED,average='macro',multi_class='ovo')
        return val_loss, acc,roc_auc


def run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val, mel_idx):

    if args.DEBUG:
        args.n_epochs = 5
        df_train = df[df['fold'] != fold].sample(args.batch_size * 5)
        df_valid = df[df['fold'] == fold].sample(args.batch_size * 5)
    else:
        df_train = df[df['fold'] != fold]
        df_valid = df[df['fold'] == fold]
  
   
    Weightedsampler_train = WeightedRandomSampler(weightedsampler(df_train.target.values).type('torch.DoubleTensor'), len(df_train.target.values))

    dataset_train = MelanomaDataset(df_train, 'train', meta_features, transform=transforms_train)
    dataset_valid = MelanomaDataset(df_valid, 'valid', meta_features, transform=transforms_val)
    #train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=RandomSampler(dataset_train), num_workers=args.num_workers)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=Weightedsampler_train, num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

    model = ModelClass(
        args.enet_type,
        n_meta_features=n_meta_features,
        n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
        out_dim=args.out_dim,
        pretrained=True,
        fusion = True,latent_dim = args.latent_dim
    )
    if DP:
        model = apex.parallel.convert_syncbn_model(model)
    model = model.to(device)
    acc_max = 0
    auc_max = 0.
    auc_20_max = 0.
    model_file  = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
    model_file2 = os.path.join(args.model_dir, f'{args.kernel_type}_best_20_fold{fold}.pth')
    model_file3 = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if DP:
        model = nn.DataParallel(model)
#     scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs - 1)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)
    
    print(len(dataset_train), len(dataset_valid))

    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), f'Fold {fold}, Epoch {epoch}')
#         scheduler_warmup.step(epoch - 1)

        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, acc, auc= val_epoch(model, valid_loader, mel_idx, is_ext=df_valid['is_ext'].values)

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, auc: {(auc):.6f}.'
        print(content)
        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content + '\n')

        scheduler_warmup.step()    
        if epoch==2: scheduler_warmup.step() # bug workaround   
        
        if acc > acc_max :
            print('acc_max ({:.6f} --> {:.6f}). Saving model ...'.format(acc_max, acc))
            torch.save(model.state_dict(), model_file)
            acc_max = acc
            
        #if auc > auc_max:
            #print('auc_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_max, auc))
            #torch.save(model.state_dict(), model_file)
            #auc_max = auc
        #if auc_20 > auc_20_max:
            #print('auc_20_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_20_max, auc_20))
            #torch.save(model.state_dict(), model_file2)
            #auc_20_max = auc_20

    #torch.save(model.state_dict(), model_file3)


def main():

    df, df_test, meta_features, n_meta_features, mel_idx = get_df(
        args.kernel_type,
        args.out_dim,
        args.data_dir,
        args.data_folder,
        args.use_meta
    )
    print("number of meta Feature",n_meta_features)
    transforms_train, transforms_val = get_transforms(args.image_size)

    folds = [int(i) for i in args.fold.split(',')]
    for fold in folds:
        run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val, mel_idx)


if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if 'efficientnet' in args.enet_type:
        ModelClass = Effnet_Melanoma
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    set_seed()

    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()

    main()
