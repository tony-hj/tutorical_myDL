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
ImageFile.LOAD_TRUNCATED_IMAGES = True
#==========================================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class p:
    def __init__(self):
        self.train_data_dir = "./dataset/train/"
        self.val_data_dir = './dataset/val'
        self.num_classes = 10

        self.batch_size = 64  

        self.EPOCH = 10

        self.pre_epoch = 0  

        self.input_size = 224

        self.outdir = './output'

param = p()
#=============================================
net = torchvision.models.resnet34(pretrained=True)
# net = EfficientNet.from_pretrained('efficientnet-b4')

net.fc.out_features = param.num_classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

# net.load_state_dict(torch.load('./model/net_035.pth'))

net = net.to(device)
params_to_update = net.parameters()
dataloaders_dict , b = get_loader(param.batch_size,param.input_size, num_workers=0)

def main():
    ii = 0
    LR = 1e-3  # 学习率
    best_acc = 0  # 初始化best test accuracy
    print("Start Training, DeepNetwork!")  # 定义遍历数据集的次数


    # criterion
    criterion = LabelSmoothSoftmaxCE()

    # optimizer
    optimizer = optim.Adam(params_to_update, lr=LR, betas=(0.9, 0.999), eps=1e-9)
    #optimizer = Ranger(net.parameters(), **kwargs)

    # scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)


    for epoch in range(param.pre_epoch, param.EPOCH):
        # scheduler.step(epoch)

        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0

        for i, data in enumerate(dataloaders_dict['train'], 0):
            # 准备数据
            length = len(dataloaders_dict['train'])
            # print(data)
            input, target = data
            input,target = input.to(device),target.to(device)
            # print(isinstance(input,Tensor)


            # 训练
            optimizer.zero_grad()
            # forward + backward
            output = net(input)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
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
            print('测试分类准确率为：%.3f%%' % (100. * float(correct) / float(total)))
            acc = 100. * float(correct) / float(total)
            scheduler.step(acc)

    torch.save(net.state_dict(), '%s/net_%03d.pth' % (param.outdir, epoch + 1))


if __name__ == "__main__":
    main()
