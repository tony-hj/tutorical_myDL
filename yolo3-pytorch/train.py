#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import os
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils.config import Config
from nets.yolo_training import YOLOLoss,Generator
from nets.yolo3 import YoloBody

def fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,genval,Epoch):
    total_loss = 0
    val_loss = 0
    for iteration in range(epoch_size):
        start_time = time.time()
        images, targets = next(gen)
        with torch.no_grad():
            images = Variable(torch.from_numpy(images).cuda().type(torch.FloatTensor))
            targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

        optimizer.zero_grad()
        outputs = net(images)
        losses = []
        for i in range(3):
            loss_item = yolo_losses[i](outputs[i], targets)
            losses.append(loss_item[0])
        loss = sum(losses)
        loss.backward()
        optimizer.step()

        total_loss += loss
        waste_time = time.time() - start_time
        print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('iter:' + str(iteration) + '/' + str(epoch_size) + ' || Total Loss: %.4f || %.4fs/step' % (total_loss/(iteration+1),waste_time))

    print('Start Validation')
    for iteration in range(epoch_size_val):
        images_val, targets_val = next(gen_val)

        with torch.no_grad():
            images_val = Variable(torch.from_numpy(images_val).cuda().type(torch.FloatTensor))
            targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
            optimizer.zero_grad()
            outputs = net(images_val)
            losses = []
            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets_val)
                losses.append(loss_item[0])
            loss = sum(losses)
            val_loss += loss
    print('Finish Validation')
    print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), '/content/drive/My Drive/yolov3/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))


if __name__ == "__main__":
    # 参数初始化
    annotation_path = '2007_train.txt'# 路径+框（4）+类别（1）
    model = YoloBody(Config)# 创建模型

    print('Loading weights into state dict...')# 载入预训练权重
    model_dict = model.state_dict()
    pretrained_dict = torch.load("model_data/yolo_weights.pth")
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model

    net = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    net = net.cuda()

    # 建立loss函数
    yolo_losses = []
    for i in range(3):
        # yolov3在预测时是分三个有效特征层进行预测的52*52 26*26 13*13
        # loss分别针对三个有效特征层
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolo"]["anchors"],[-1,2]),
                                    Config["yolo"]["classes"], (Config["img_w"], Config["img_h"])))
    # 验证集和训练集的划分
    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    if True:
        lr = 1e-4
        Batch_size = 8
        Init_Epoch = 0
        Freeze_Epoch = 25# 冻结模型持续的epoch 
        
        optimizer = optim.Adam(net.parameters(),lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
        
        # 得到生成器
        gen = Generator(Batch_size, lines[:num_train],
                        (Config["img_h"], Config["img_w"])).generate()
        gen_val = Generator(Batch_size, lines[num_train:],
                        (Config["img_h"], Config["img_w"])).generate()
                        
        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        # 开始训练
        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch)
            lr_scheduler.step()

    if True:
        # 超参数
        lr = 1e-5
        Batch_size = 4
        Freeze_Epoch = 25
        Unfreeze_Epoch = 50
        
        
        optimizer = optim.Adam(net.parameters(),lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
        gen = Generator(Batch_size, lines[:num_train],
                        (Config["img_h"], Config["img_w"])).generate()
        gen_val = Generator(Batch_size, lines[num_train:],
                        (Config["img_h"], Config["img_w"])).generate()
                        
        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch)
            lr_scheduler.step()
