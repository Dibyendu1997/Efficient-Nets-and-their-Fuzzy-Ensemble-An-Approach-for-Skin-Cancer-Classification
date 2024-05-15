import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import cv2
import PIL.Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import GradualWarmupSchedulerV2
import apex
#from apex import amp
from dataset import get_df, get_transforms, MelanomaDataset
from models import Effnet_Melanoma, Resnest_Melanoma, Seresnext_Melanoma,ViTBase16
from train import get_trans
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from choquet_fuzzy import ensemble

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='/raid/')
    parser.add_argument('--data-folder', type=int, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    #parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--out-dim', type=int, default=8)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    #parser.add_argument('--model-dir', type=str, default='./weights_with_hair_b4')
    #parser.add_argument('--model-dir', type=str,choices=['./weights_with_hair_b4','./weights_with_hair_b5','./weights_with_hair'])
    parser.add_argument('--log-dir', type=str, default='./logs_with_hair_b4')
    parser.add_argument('--sub-dir', type=str, default='./subs')
    parser.add_argument('--eval', type=str, choices=['best', 'best_20', 'final'], default="best")
    parser.add_argument('--n-test', type=int, default=8)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')

    args, _ = parser.parse_known_args()
    return args


def main():

    df, df_test, meta_features, n_meta_features, mel_idx = get_df(
        "8c_meta128_32_b4ns_384_ext_50ep",
        args.out_dim,
        args.data_dir,
        args.data_folder,
        args.use_meta
    )
    df_test = df_test.iloc[:2]

    transforms_train, transforms_val = get_transforms(args.image_size)

    if args.DEBUG:
        df_test = df_test.sample(args.batch_size * 3)
    dataset_test = MelanomaDataset(df_test, 'test', meta_features, transform=transforms_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers)
    dir1 = './weights_sres101'
    dir2 = './weights_with_hair_b5'
    dir3 = './weights_with_hair'
    enet_type1 = 'seresnext101'
    enet_type2 = 'tf_efficientnet_b5_ns'
    enet_type3 = 'tf_efficientnet_b6_ns'
    kernel_type1 = "8c_meta128_32_b5ns_384_ext_50ep"
    kernel_type2 = "8c_meta128_32_b5ns_384_ext_50ep"
    kernel_type3 = "8c_meta128_32_b5ns_384_ext_50ep"

    # load model
    models1 = []
    models2 = []
    models3 = []
    b4 = list()
    b5 = list ()
    b6 = list()
    for fold in range(5):
        if args.eval == 'best':
            model_file1 = os.path.join(dir1, f'{kernel_type1}_best_fold{fold}.pth')
            model_file2 = os.path.join(dir2, f'{kernel_type2}_best_fold{fold}.pth')
            model_file3 = os.path.join(dir3, f'{kernel_type3}_best_fold{fold}.pth')
            print("Model File",model_file1)
            print("Model File",model_file2)
            print("Model File",model_file3)

        model1 = ModelClass1(
            enet_type1,
            n_meta_features=n_meta_features,
            n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
            out_dim=args.out_dim
        )
        model1 = model1.to(device)
        
        
        
        model2 = ModelClass(
            enet_type2,
            n_meta_features=n_meta_features,
            n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
            out_dim=args.out_dim
        )
        model2 = model2.to(device)
        
        
        model3 = ModelClass(
            enet_type3,
            n_meta_features=n_meta_features,
            n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
            out_dim=args.out_dim
        )
        model3 = model3.to(device)
        
        
        try:  # single GPU model_file
            model1.load_state_dict(torch.load(model_file1), strict=True)
            model2.load_state_dict(torch.load(model_file2), strict=True)
            model3.load_state_dict(torch.load(model_file3), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)
        
        if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
            model = torch.nn.DataParallel(model)

        model1.eval()
        models1.append(model1)
        
        
        model2.eval()
        models2.append(model2)
        
        
        model3.eval()
        models3.append(model3)
        #print("A")      
        
        
        
    #print("printing Models",models)

    # predict
    PROBS1 = []
    PROBS2 = []
    PROBS3 = []
    TARGETS = []
    with torch.no_grad():
        for (data,target) in tqdm(test_loader):
            if args.use_meta:
                data, meta = data
                data, meta ,target= data.to(device), meta.to(device) ,target.to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for model in models:
                	l = model(data, meta)
                	probs += l.softmax(1)
                    #for I in range(args.n_test):
                        #print(get_trans(data, I))
                        #print(data)
                        #l = model(get_trans(data, I), meta)
                        #print("output shape",l.shape)
                        #probs += l.softmax(1)
            else:   
                data ,target = data.to(device) , target.to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs1 = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs2 = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs3 = torch.zeros((data.shape[0], args.out_dim)).to(device)                
                for model in models1:
                	l = model(data)
                	#print(l.softmax(1).shape)
                	probs1 += l.softmax(1)
                for model in models2:
                	l = model(data)
                	#print(l.softmax(1).shape)
                	probs2 += l.softmax(1)
                for model in models3:
                	l = model(data)
                	#print(l.softmax(1).shape)
                	probs3 += l.softmax(1)
            #probs /= args.n_test
            probs1 /= len(models1)
            #print("*****",probs1[0])
            probs2 /= len(models2)
            probs3 /= len(models3)
            b4.append(probs1[0])
            b5.append(probs2[0])
            b6.append(probs3[0])
            #print("a",probs.shape)

            PROBS1.append(probs1.detach().cpu())
            PROBS2.append(probs2.detach().cpu())
            PROBS3.append(probs3.detach().cpu())
            TARGETS.append(target.detach().cpu())

    PROBS1 = torch.cat(PROBS1).numpy()
    PROBS2 = torch.cat(PROBS2).numpy()
    PROBS3 = torch.cat(PROBS3).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    acc1 = (PROBS1.argmax(1) == TARGETS).mean() * 100.
    acc2 = (PROBS2.argmax(1) == TARGETS).mean() * 100.
    acc3 = (PROBS3.argmax(1) == TARGETS).mean() * 100.
    print(acc1,acc2,acc3)
    s = sum([acc1,acc2,acc3])
    measures = [acc1/s,acc2/s]
    #y_pred = PROBS.argmax(1)
    sugeno_predictions = list()
    for i in range(len(df_test)):
    	pred = ensemble([b4[i], b5[i]], measures,mode='choquet')
    	sugeno_predictions.append(pred)


    print("Classification Report Fuzzy",classification_report(TARGETS, sugeno_predictions))
    print("Balanced Accuracy score for isic comparision Fuzzy",balanced_accuracy_score(TARGETS, sugeno_predictions))


if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.sub_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    ModelClass = Effnet_Melanoma
    ModelClass1 = Seresnext_Melanoma

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    device = torch.device('cuda')

    main()
