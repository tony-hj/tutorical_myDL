# -*- coding: utf-8 -*-
from torchvision import datasets, transforms
import torch
import torchvision
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from efficientnet_pytorch import EfficientNet
from utils.label_smooth import LabelSmoothSoftmaxCE
import utils.config as config
from utils.dataloader import get_debug_loader
from utils.ranger import Ranger
import os
from PIL import ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

from mean_std import calc_mean_std
torch.manual_seed(123)            # 为CPU设置随机种子
torch.cuda.manual_seed(123)       # 为当前GPU设置随机种子
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if config.model_path:
    net = EfficientNet.from_pretrained('efficientnet-b4',weights_path=config.model_path,num_classes=config.num_classes)
else:
    net = EfficientNet.from_pretrained('efficientnet-b4',num_classes=config.num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)


if not os.path.exists(config.outdir):
    os.mkdir(config.outdir)

mean, std = calc_mean_std() # 会自动打印，由你决定改不改

net = net.to(device)
dataloaders_dict, cls2id = get_debug_loader(config.root)

criterion = LabelSmoothSoftmaxCE() if config.label_smooth else nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=config.LR, betas=(0.9, 0.999), eps=1e-9)
# optimizer = Ranger(net.parameters(), lr=config.LR)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=Config.milestone, gamma=0.1)

def train(criterion,optimizer,scheduler,LR=config.LR,debug=False):

    val_accs = [0]
    train_losses = []
    best_acc = 0  
    bad_data = []

    for epoch in range(config.epochs):

        net.train()
        epoch_loss = 0.0
        correct = 0.0
        total = 0.0
        batchs = len(dataloaders_dict['train'])
        
        with tqdm(total=batchs) as pbar:
            pbar.set_description(f"Train Epoch {epoch + 1} / {config.epochs}")

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
                
                if debug:
                    images, labels, paths = data
                else:
                    images, labels, _ = data
                images, labels = images.to(device), labels.to(device)
                
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).cpu().sum()
                if debug: # 如果训练效果不佳可以返回每个epoch里面错误的数据
                    bad_data_one_epoch.append([paths[predicted == labels],labels[predicted == labels]])
            
            bad_data.append(bad_data_one_epoch)    
            acc = 100. * float(correct) / float(total)     
            
            scheduler.step(acc)
            
            print('\t测试分类准确率为：%.3f%%' % acc)

            if acc > max(val_accs):
                print("\tsaving best model so far")
                torch.save(net.state_dict(), '%s/net_%03d_%.3f.pth' % (config.outdir, epoch + 1,acc))

            val_accs.append(acc)
        

    if not os.path.exists(config.outdir):
        os.mkdir(config.outdir)
    torch.save(net.state_dict(), '%s/net_%03d_%.3f.pth' % (config.outdir, epoch + 1,acc))
    if debug:
        return bad_data # [[path&class],[],[],[]]


train(criterion,optimizer,scheduler)
