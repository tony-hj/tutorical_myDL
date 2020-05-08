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
from dataloader import get_loader
from PIL import ImageFile
from resnet import resnet50
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True
#==========================================
torch.manual_seed(123)            # 为CPU设置随机种子
torch.cuda.manual_seed(123)       # 为当前GPU设置随机种子
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class p:
    def __init__(self):
        self.train_data_dir = "./dataset/train/"
        self.val_data_dir = './dataset/val'
        
        self.num_classes = 2

        self.batch_size = 40

        self.EPOCH = 100

        self.pre_epoch = 0

        self.pretrained_path = ''

        self.input_size = 380

        self.outdir = './output'

param = p()
#=============================================
# net = resnet50(pretrained=False)
# mydict = torch.load('/root/.cache/torch/checkpoints/resnet50-19c8e357.pth')
# print('loading...')
# for each in tqdm(mydict.keys()):
  # resnet50().state_dict()[each] = mydict[each]
# print('successful!')
net = EfficientNet.from_pretrained('efficientnet-b4')

net._fc.out_features = param.num_classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

if param.pre_epoch > 0 :
    net.load_state_dict(torch.load(param.pretrained_path))

net = net.to(device)
params_to_update = net.parameters()
dataloaders_dict , b = get_loader(param.batch_size,param.input_size, num_workers=0)

def train():
    ii = 0
    LR = 1e-3  
    val_accs = []
    train_losses = []
    best_acc = 0  
    
    # 损失函数 优化器 scheduler
    criterion = LabelSmoothSoftmaxCE()
    optimizer = optim.Adam(params_to_update, lr=LR, betas=(0.9, 0.999), eps=1e-9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)


    for epoch in range(param.pre_epoch, param.EPOCH):

        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0

        for i, data in enumerate(dataloaders_dict['train'], 0):

            length = len(dataloaders_dict['train']) # 每个epoch有多少个batch
            input, target = data
            input,target = input.to(device),target.to(device)

            # 训练
            optimizer.zero_grad()
            # forward + backward
            output = net(input)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            _, predicted = torch.max(output.data, 1)
            sum_loss += loss.item()
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            train_losses.append(sum_loss / (i + 1))
            
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                     100. * float(correct) / float(total)))
            
        # 每训练完一个epoch测试一下准确率
        print("Waiting Test!")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in dataloaders_dict['val']:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).cpu().sum()
            acc = 100. * float(correct) / float(total)
            
            print('测试分类准确率为：%.3f%%' % acc)
            if acc > max(val_accs):
                print("高准确率，保存模型！")
                torch.save(net.state_dict(), '%s/net_%03d_%.3f.pth' % (param.outdir, epoch + 1,acc))
            val_accs.append(acc)
            scheduler.step(acc)

    torch.save(net.state_dict(), '%s/net_%03d_%.3f.pth' % (param.outdir, epoch + 1,acc))

if __name__ == "__main__":
    train()
