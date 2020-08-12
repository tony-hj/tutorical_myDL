from torchvision import transforms
import numpy as np
import torch
from PIL import Image
import torchvision

class Resize_propotion(object):
    """
    等比缩放，防止在缩放过程中导致物体长宽比例改变导致识别错误
    """
    def __init__(self,size,interpolation = Image.BILINEAR):
        self.size = size
        self.interpolation =interpolation

    def __call__(self,img):
        #padding
        ratio = self.size[0] / self.size[1]
        w,h = img.size
        if w/h < ratio:
            t = int(h * ratio)
            w_padding = (t-w)//2
            img = img.crop((-w_padding,0,w+w_padding,h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))

        img = img.resize(self.size, self.interpolation)
        return img

input_size = 380

mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2470, 0.2435, 0.2616])

trans_tta_1 = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
trans_tta_2 = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
trans_tta_3 = transforms.Compose([
    transforms.Resize(input_size),
    transforms.RandomRotation((-25, 25)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
trans_tta_4 = transforms.Compose([
    transforms.Resize(input_size),
    transforms.RandomHorizontalFlip(0.25),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
trans_tta_5 = transforms.Compose([
    transforms.Resize(input_size),
    transforms.RandomVerticalFlip(0.25),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
trans_tta_6 = transforms.Compose([
    Resize_propotion(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

tta_trans_list = [trans_tta_1,trans_tta_2,trans_tta_3,trans_tta_4,trans_tta_5,trans_tta_6]
    
test_transform = transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.CenterCrop(input_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])

train_transform = transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.CenterCrop(input_size),
    torchvision.transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])

batch_size = 20
num_workers = 4
label_smooth = True
LR = 1e-3

# 下面的参数一般要改

num_classes = 20
root = './' # 传给data_pps的参数
epochs = 20
model_path = '' # 预训练模型的位置
outdir = './' # 路径后面不能有斜杠   '%s/net_%03d_%.3f.pth' % (config.outdir, epoch + 1,acc))





