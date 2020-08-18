# -*- coding: utf-8 -*-
from torchvision import datasets, transforms
import torch
import torchvision
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from efficientnet_pytorch import EfficientNet, cbam_EfficientNet

from utils.label_smooth import LabelSmoothSoftmaxCE
import utils.config as cfg
from utils.dataloader import get_debug_loader
from utils.ranger import Ranger
from utils.data_pps import get_lists

import os
import argparse
import pandas as pd
from PIL import ImageFile, Image
from tqdm import tqdm
import numpy as np
from mean_std import calc_mean_std
from torch.utils.model_zoo import load_url

def test(net, test_dict, idx):
    
    
    test_path = test_dict['paths']
    test_label = test_dict['labels']
    
    ids = [i.split('/')[-1][:-4] for i in test_dict['paths']]
    res = []
    
    for i in range(len(test_path)):
        filename = test_path[i]
        image = Image.open(os.path.join(cfg.root,filename)).convert('RGB')
        input = cfg.test_transform(image).unsqueeze(0).to(device)
        output = net(input)
        label = int(output.argmax())
        res.append(label)

    final_res = [[ids[i], res[i]] for i in range(len(ids))]
    df = pd.DataFrame(final_res, columns=['FileID', 'SpeciesID'])
    df.to_csv('id_{}.csv'.format(idx),index=None)
    acc = sum(np.array(res) == np.array(test_label)) / len(res)
    print('bagging {} : {}'.format(idx, acc))
    print('over!')
    
def init_net(cfg):

    net = cbam_EfficientNet.from_pretrained('efficientnet-b4',weights_path=cfg.pre_model,num_classes=cfg.num_classes,cbam=cfg.cbam)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net) 
        
    net = net.to(device)
    
    return net


def train(net, criterion, optimizer, scheduler, dataloaders_dict, cfg):

    val_accs = [0]
    train_losses = []
    best_acc = 0  
    bad_data = []

    for epoch in range(cfg.epochs):

        net.train()
        epoch_loss = 0.0
        correct = 0.0
        total = 0.0
        batchs = len(dataloaders_dict['train'])
        
        with tqdm(total=batchs) as pbar:
            pbar.set_description(f"Train Epoch {epoch + 1} / {cfg.epochs}")

            for batch_idx, data in enumerate(dataloaders_dict['train'], 0):

                input, target, _ = data
                input,target = input.to(device),target.to(device)

                # 训练
                optimizer.zero_grad()
                output = net(input)
                batch_loss = criterion(output, target)

                batch_loss.backward()
                optimizer.step()

                # 每训练1个batch打印一次loss和准确率
                _, predicted = torch.max(output.data, 1)
                epoch_loss += batch_loss.item()
                total += target.size(0)
                correct += predicted.eq(target.data).cpu().sum()
                train_losses.append(epoch_loss / (batch_idx + 1))
                
                pbar.set_postfix({'iter':           (batch_idx + 1 + epoch * batchs), 
                                  'Epoch_Avg_Loss': epoch_loss / (batch_idx + 1), 
                                  'Acc':            100. * float(correct) / float(total)})
                pbar.update(1)
                
        # 每训练完一个epoch测试一下准确率
        with torch.no_grad():
            bad_data_one_epoch = []
            correct = 0
            total = 0
            for data in dataloaders_dict['val']:
                net.eval()
                
                if cfg.debug:
                    images, labels, paths = data
                else:
                    images, labels, _ = data
                images, labels = images.to(device), labels.to(device)
                
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).cpu().sum()
                if cfg.debug: # 如果训练效果不佳可以返回每个epoch里面错误的数据
                    bad_data_one_epoch.append([paths[predicted == labels],labels[predicted == labels]])
            
            bad_data.append(bad_data_one_epoch)    
            acc = 100. * float(correct) / float(total)     
            
            scheduler.step(acc)
            
            print('\t测试分类准确率为：%.3f%%' % acc)

            if acc > max(val_accs):
                print("\tsaving best model so far")
                torch.save(net.state_dict(), '%s/net_%03d_%.3f.pth' % (cfg.out_dir, epoch + 1,acc))

            val_accs.append(acc)
        
    torch.save(net.state_dict(), '%s/net_%03d_%.3f.pth' % (cfg.out_dir, epoch + 1,acc))
    if cfg.debug:
        return bad_data # [[path&class],[],[],[]]


def concat_res(test_dict):

    def get_common(res):
        x = dict((a,res.count(a)) for a in res)
        cls = [k for k,v in x.items() if max(x.values())==v][0]
        return cls
        
    paths = [i.split('/')[-1][:-4] for i in test_dict['paths']]
    labels = test_dict['labels']
    
    res_dict = {'FileID':paths, 'SpeciesID':[[] for i in range(len(paths))]}
    for dir in ['id_0.csv','id_1.csv','id_2.csv','id_3.csv','id_4.csv']:
        df = pd.read_csv(dir)
        for i in range(len(df)):
            key = df.loc[i,:]['FileID']
            value = df.loc[i,:]['SpeciesID']
            res_dict['SpeciesID'][i].append(value)
            
    for i in range(len(df)):
        res_dict['SpeciesID'][i] = get_common(res_dict['SpeciesID'][i])
        
    final_res = [[res_dict['FileID'][i], res_dict['SpeciesID'][i]] for i in range(len(paths))]
    df = pd.DataFrame(final_res, columns=['FileID', 'SpeciesID'])
    df.to_csv('bagging_res.csv',index=None)
    acc = sum(np.array(res_dict['SpeciesID']) == np.array(labels)) / len(paths)
    print('final bagging acc is ',acc) 
    


if __name__ == '__main__':
    
    torch.manual_seed(123)            # 为CPU设置随机种子
    torch.cuda.manual_seed(123)       # 为当前GPU设置随机种子
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(cfg.out_dir):
        os.mkdir(cfg.out_dir)
        
    if cfg.mean_std:
        mean, std = calc_mean_std() # 会自动打印，由你决定改不改

    
    if cfg.bagging:
        paths, labels, _ =  get_lists(cfg.root)
        test_dict = {'paths':paths['test'], 'labels':labels['test']}

        for idx in range(5):
            print('bagging iter {}'.format(idx))
            net = init_net(cfg)
            criterion = LabelSmoothSoftmaxCE() if cfg.label_smooth else nn.CrossEntropyLoss().to(device)
            optimizer = Ranger(net.parameters(), lr=cfg.lr) # optim.Adam()
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True) # optim.lr_scheduler.MultiStepLR
            dataloaders_dict, cls2id = get_debug_loader(cfg.root, idx)
            train(net,criterion,optimizer,scheduler,dataloaders_dict,cfg)
            test(net, test_dict, idx)
            del net
            
        concat_res(test_dict)

    else:
        net = init_net(cfg)
        criterion = LabelSmoothSoftmaxCE() if cfg.label_smooth else nn.CrossEntropyLoss().to(device)
        optimizer = Ranger(net.parameters(), lr=cfg.lr) # optim.Adam()
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True) # optim.lr_scheduler.MultiStepLR
        dataloaders_dict, cls2id = get_debug_loader(cfg.root)
        train(net,criterion,optimizer,scheduler,dataloaders_dict,cfg)
            
