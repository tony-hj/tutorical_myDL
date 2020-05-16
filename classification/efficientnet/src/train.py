# -*- coding: utf-8 -*-
from torchvision import datasets, transforms
import torch
import torchvision
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from efficientnet_pytorch import EfficientNet
from label_smooth import LabelSmoothSoftmaxCE
import os
from utils.dataloader import get_debug_loader
from PIL import ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True
import utils.config as config 
#==========================================
torch.manual_seed(123)            # 为CPU设置随机种子
torch.cuda.manual_seed(123)       # 为当前GPU设置随机种子
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


net = EfficientNet.from_pretrained('efficientnet-b4')

net._fc.out_features = num_classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

if model_path:
    net.load_state_dict(torch.load(config.model_path))

net = net.to(device)
params_to_update = net.parameters()
dataloaders_dict, cls2id = get_debug_loader(type=1,merge=True)

criterion = LabelSmoothSoftmaxCE() if config.label_smooth else nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(params_to_update, lr=LR, betas=(0.9, 0.999), eps=1e-9)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=Config.milestone, gamma=0.1)

def train(LR = config.LR,criterion,optimizer,scheduler,debug=False):

    val_accs = []
    train_losses = []
    best_acc = 0  
    bad_data = []
    
    for epoch in range(pre_epochs, config.epochs):

        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        epoch_loss = 0.0
        correct = 0.0
        total = 0.0

        for batch_idx, data in enumerate(dataloaders_dict['train'], 0):

            batchs = len(dataloaders_dict['train']) # 每个epoch有多少个batch
            input, target = data
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
            
            print('[epoch:%d, iter:%d] Epoch_Avg_Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (batch_idx + 1 + epoch * batchs), epoch_loss / (batch_idx + 1),
                     100. * float(correct) / float(total)))
            
        # 每训练完一个epoch测试一下准确率
        print("Waiting Test!")
        with torch.no_grad():
            bad_data_one_epoch = []
            correct = 0
            total = 0
            for data in dataloaders_dict['val']:
                net.eval()
                
                if debug:
                    images, labels,paths = data
                else:
                    images, labels = data
                images, labels = images.to(device), labels.to(device)
                
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).cpu().sum()
                if debug: # 如果训练效果不佳可以返回每个epoch里面错误的数据
                    bad_data_one_epoch.append([paths[predicted == labels],labels[predicted == labels]])
            
            bad_data.append(bad_data_one_epoch)    
            acc = 100. * float(correct) / float(total)     
            val_accs.append(acc)
            scheduler.step(acc)
            
            print('测试分类准确率为：%.3f%%' % acc)
            
            if acc > max(val_accs):
                print("高准确率，保存模型！")
                torch.save(net.state_dict(), '%s/net_%03d_%.3f.pth' % (config.outdir, epoch + 1,acc))
            


    torch.save(net.state_dict(), '%s/net_%03d_%.3f.pth' % (config.outdir, epoch + 1,acc))
    if debug:
        return bad_data # [[path&class],[],[],[]]


train(criterion,optimizer,scheduler)
