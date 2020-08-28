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
import seaborn as sns
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageFile, Image
from tqdm import tqdm
import numpy as np
from mean_std import calc_mean_std
from torch.utils.model_zoo import load_url


def confusion_matrix(a,idx=-1,tta=False):
    # 删除准确率为1的行/列
    row=0
    while row < len(a):
        if a[row,row] == sum(a[row]) and a[row,row] == sum(a[:,row]):
            a = np.delete(a,row,1)
            a = np.delete(a,row,0)
            row -= 1
        row += 1
    plt.clf()
    sns.heatmap(a,annot=True,cmap='YlGnBu',annot_kws={'size':10,'weight':'bold'})
    plt.tick_params(labelsize=10)
    plt.ylabel('prediction',fontsize=15)
    plt.xlabel('ground-truth',fontsize=15)
    fig = plt.gcf()
    fig.savefig("tta_conf_matrix_{}.png".format(idx) if tta else "conf_matrix_{}.png".format(idx))


def test(net, idx, tta=False, opt=False):
    paths, labels, _ =  get_lists(cfg.root,opt)
    test_dict = {'paths':paths['test'], 'labels':labels['test']}
    
    test_path = test_dict['paths']
    test_label = test_dict['labels']
    
    ids = [i.split('/')[-1][:-4] for i in test_dict['paths']]
    res = []
    
    for i in range(len(test_path)):
        filename = test_path[i]
        image = Image.open(os.path.join(cfg.root,filename)).convert('RGB')
        if tta:
            images = []
            for tsfm in cfg.tta_trans_list:
                images.append(tsfm(image).unsqueeze(0).to(device))
            output = [net(i) for i in images]
            label0 = [int(i.argmax()) for i in output]

            b = {i:sum([j==i for j in label0]) for i in set(label0)}
            a = np.array([i for i in b.values()])
            label = list(b.keys())[a.argmax()]
            if len([i for i in a if i == max(a)]) > 1:
                print('you need get more tta tsfm',label0)
            del images, output

        else:
            input = cfg.test_transform(image).unsqueeze(0).to(device)
            output = net(input)
            label = int(output.argmax())
        res.append(label)
    
    if cfg.confusion_matrix:
        conf_matrix = np.zeros((cfg.num_classes,cfg.num_classes))
        for p, t in zip(res, test_label):
            conf_matrix[p,t] += 1
            
        confusion_matrix(conf_matrix,idx,tta)
        

    
    final_res = [[ids[i], res[i]] for i in range(len(ids))]
    df = pd.DataFrame(final_res, columns=['FileID', 'SpeciesID'])
    df.to_csv('tta_id_{}.csv'.format(idx) if tta else 'id_{}.csv'.format(idx),index=None)
    acc = sum(np.array(res) == np.array(test_label)) / len(res)
    print('bagging {} : {}'.format(idx, acc))
    print('over!')
   
   
def init_net(cfg,v=4):

    net = cbam_EfficientNet.from_pretrained('efficientnet-b{}'.format(v),weights_path=cfg.pre_model,num_classes=cfg.num_classes,cbam=cfg.cbam)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net) 
        
    net = net.to(device)
    
    return net


def concat_res(tta=False,opt=False):

    def get_common(res):
        x = dict((a,res.count(a)) for a in res)
        cls = [k for k,v in x.items() if max(x.values())==v][0]
        return cls
        
    paths, labels, _ =  get_lists(cfg.root,opt)
    test_dict = {'paths':paths['test'], 'labels':labels['test']}
    
    paths = [i.split('/')[-1][:-4] for i in test_dict['paths']]
    labels = test_dict['labels']
    
    res_dict = {'FileID':paths, 'SpeciesID':[[] for i in range(len(paths))]}
    for dir in ['tta_id_0.csv','tta_id_1.csv','tta_id_2.csv','tta_id_3.csv','tta_id_4.csv'] if tta else ['id_0.csv','id_1.csv','id_2.csv','id_3.csv','id_4.csv']:
        df = pd.read_csv(dir)
        for i in range(len(df)):
            key = df.loc[i,:]['FileID']
            value = df.loc[i,:]['SpeciesID']
            res_dict['SpeciesID'][i].append(value)
            
    for i in range(len(df)):
        res_dict['SpeciesID'][i] = get_common(res_dict['SpeciesID'][i])
        
    if cfg.confusion_matrix:
        conf_matrix = np.zeros((cfg.num_classes,cfg.num_classes))
        for p, t in zip(res_dict['SpeciesID'], labels):
            conf_matrix[p,t] += 1
            
        confusion_matrix(conf_matrix)    
        
    final_res = [[res_dict['FileID'][i], res_dict['SpeciesID'][i]] for i in range(len(paths))]
    df = pd.DataFrame(final_res, columns=['FileID', 'SpeciesID'])
    df.to_csv('tta_bagging_res.csv' if tta else 'bagging_res.csv',index=None)
    acc = sum(np.array(res_dict['SpeciesID']) == np.array(labels)) / len(paths)
    if tta:
        print('final tta bagging acc is ',acc)
    else:
        print('final bagging acc is ',acc)
    
def cat_res(path0, path20):
    df0 = pd.read_csv(path0)
    df20 = pd.read_csv(path20)
    index = []
    for idx in range(len(df0)):
        if df0.iloc[idx,1] == 0:
            index.append(idx)
    for idx in index:
        df20.iloc[idx,1] = 0
        
    df20.to_csv('final_res.csv')
  
def get_acc(path):
    df_test = pd.read_csv('/content/dataset/test.csv')
    df = pd.read_csv(path)
    acc = sum(np.array(df['SpeciesID']) == np.array(df_test['SpeciesID'])) / len(df)
    
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
            
            print('\t验证集分类准确率为：%.3f%%' % acc)
            
            if cfg.save:
                if acc > max(val_accs):
                    print("\tsaving best model so far")
                    torch.save(net.state_dict(), '%s/net_%03d_%.3f.pth' % (cfg.out_dir, epoch + 1,acc))

            val_accs.append(acc)
              
            
    if cfg.debug:
        return bad_data # [[path&class],[],[],[]]



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

        for idx in range(5):
            print('bagging iter {}'.format(idx))
            net = init_net(cfg)
            criterion = LabelSmoothSoftmaxCE() if cfg.label_smooth else nn.CrossEntropyLoss().to(device)
            optimizer = Ranger(net.parameters(), lr=cfg.lr) # optim.Adam()
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True) # optim.lr_scheduler.MultiStepLR
            dataloaders_dict, cls2id = get_debug_loader(cfg.root, idx)
            train(net,criterion,optimizer,scheduler,dataloaders_dict,cfg)
            test(net, idx)
            if cfg.tta:
                test(net, idx, cfg.tta)
            del net
            
        concat_res()
        concat_res(tta=False)

    else:
        net = init_net(cfg)
        criterion = LabelSmoothSoftmaxCE() if cfg.label_smooth else nn.CrossEntropyLoss().to(device)
        optimizer = Ranger(net.parameters(), lr=cfg.lr) # optim.Adam()
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True) # optim.lr_scheduler.MultiStepLR
        dataloaders_dict, cls2id = get_debug_loader(cfg.root)
        train(net,criterion,optimizer,scheduler,dataloaders_dict,cfg)
        test(net, 'full', cfg.tta)
        del net, dataloaders_dict, cls2id
        # 训练一个helper_net
        helper_net = init_net(cfg)
        dataloaders_dict, cls2id = get_debug_loader(cfg.root,opt=True)
        train(helper_net,criterion,optimizer,scheduler,dataloaders_dict,cfg)
        test(net, 'help', cfg.tta, opt=True)
    
    
