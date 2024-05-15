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
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='/raid/')
    parser.add_argument('--data-folder', type=int, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--out-dim', type=int, default=8)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights_with_hair')
    parser.add_argument('--log-dir', type=str, default='./weights_with_hair')
    parser.add_argument('--sub-dir', type=str, default='./subs')
    parser.add_argument('--eval', type=str,default="best")
    parser.add_argument('--n-test', type=int, default=8)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')

    args, _ = parser.parse_known_args()
    return args
    
    

   
    
    

def main():

    df, df_test, meta_features, n_meta_features, mel_idx = get_df(
        args.kernel_type,
        args.out_dim,
        args.data_dir,
        args.data_folder,
        args.use_meta
    )

    transforms_train, transforms_val = get_transforms(args.image_size)
    #df_test = df_test.iloc[:5]

    if args.DEBUG:
        df_test = df_test.sample(args.batch_size * 3)
    dataset_test = MelanomaDataset(df_test, 'test', meta_features, transform=transforms_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers)

    # load model
    models = []
    for fold in range(5):

        if args.eval == 'best':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
            print("Model File",model_file)

        model = ModelClass(
            args.enet_type,
            n_meta_features=n_meta_features,
            n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
            out_dim=args.out_dim
        )
        model = model.to(device)

        try:  # single GPU model_file
            model.load_state_dict(torch.load(model_file), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)
        
        if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
            model = torch.nn.DataParallel(model)

        model.eval()
        models.append(model)
        
        
        
    # predict
    PROBS = []
    PROBS0 = []
    PROBS1 = []
    PROBS2 = []
    PROBS3 = []
    PROBS4 = []
    TARGETS = []
    fold0 = list()
    fold1 = list()
    fold2 = list()
    fold3 = list()
    fold4 = list()
    with torch.no_grad():
        for (data,target) in tqdm(test_loader):
            if args.use_meta:
                data, meta = data
                data, meta ,target= data.to(device), meta.to(device) ,target.to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs0 = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs1 = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs2 = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs3 = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs4 = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for model in models:
                	l = model(data, meta)
                	print(l.shape)
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
                probs0 = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs1 = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs2 = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs3 = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs4 = torch.zeros((data.shape[0], args.out_dim)).to(device)
                model0 ,model1 ,model2 ,model3 ,model4 = models[0],models[1],models[2],models[3],models[4]
                fold0.append(model0(data)[0])
                fold1.append(model1(data)[0])
                fold2.append(model2(data)[0])
                fold3.append(model3(data)[0])
                fold4.append(model4(data)[0])
                
                l0 = model0(data)
                probs0 += l0.softmax(1)
                
                l1 = model1(data)
                probs1 += l1.softmax(1)
                
                l2 = model2(data)
                probs2 += l2.softmax(1)
                
                l3 = model3(data)
                probs3 += l3.softmax(1)
                
                l4 = model4(data)
                probs4 += l4.softmax(1)
                
                for model in models:
                	l = model(data)
                	#print(l[0])
                	probs += l.softmax(1)
                	#for I in range(args.n_test):
                		#l = model(get_trans(data, I))
                		#probs += l.softmax(1)

            #probs /= args.n_test
            probs /= len(models)

            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())
            PROBS0.append(probs0.detach().cpu())
            PROBS1.append(probs1.detach().cpu())
            PROBS2.append(probs2.detach().cpu())
            PROBS3.append(probs3.detach().cpu())
            PROBS4.append(probs4.detach().cpu())            
            
            
    PROBS0 = torch.cat(PROBS0).numpy()
    PROBS1 = torch.cat(PROBS1).numpy()
    PROBS2 = torch.cat(PROBS2).numpy()
    PROBS3 = torch.cat(PROBS3).numpy()
    PROBS4 = torch.cat(PROBS4).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    acc0 = (PROBS0.argmax(1) == TARGETS).mean() * 100.
    acc1 = (PROBS1.argmax(1) == TARGETS).mean() * 100.
    acc2 = (PROBS2.argmax(1) == TARGETS).mean() * 100.
    acc3 = (PROBS3.argmax(1) == TARGETS).mean() * 100.
    acc4 = (PROBS4.argmax(1) == TARGETS).mean() * 100.
    print("Five Fold Accuracy",acc0,acc1,acc2,acc3,acc4)   
    print(len(fold0))
    print(len(fold1))
    s = sum([acc0, acc1,acc2,acc3,acc4]) 
    #print(model0.parameters())
    #measures = [acc0/s, acc1/s,acc2/s,acc3/s,acc4/s]
    measures = [.95, .8,0.9,0.78,0.82]
    sugeno_predictions = list()
    for i in range(len(df_test)):
    	pred = ensemble([fold0[i], fold1[i], fold2[i],fold3[i],fold4[i]], measures)
    	sugeno_predictions.append(pred)
    #print(sugeno_predictions)
    #print(classification_report(testGen.classes,sugeno_predictions, target_names=testGen.class_indices, digits=3))
    PROBS = torch.cat(PROBS).numpy()
    #TARGETS = torch.cat(TARGETS).numpy()
    acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
    y_pred = PROBS.argmax(1)

    #print("prob mel_idx",PROBS[:,mel_idx])
    #print("prob mel idx",PROBS[:,mel_idx].shape)
    print("Accuracy for test is ------>",acc)
    print("Classification Report",classification_report(TARGETS, y_pred))
    print("Classification Report Fuzzy",classification_report(TARGETS, sugeno_predictions))
    print("Balanced Accuracy score for isic comparision",balanced_accuracy_score(TARGETS, y_pred))
    print("Balanced Accuracy score for isic comparision Fuzzy",balanced_accuracy_score(TARGETS, sugeno_predictions))
    
    
if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.sub_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if args.enet_type == 'resnest101':
        ModelClass = Resnest_Melanoma
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext_Melanoma
    elif args.enet_type == 'vit_base_patch16_224':
        ModelClass = ViTBase16
    elif 'efficientnet' in args.enet_type:
        ModelClass = Effnet_Melanoma
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    device = torch.device('cuda')

    main()

