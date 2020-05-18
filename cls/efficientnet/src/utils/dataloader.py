import numpy as np
import torch
import torchvision
import os
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from prefetch_generator import BackgroundGenerator
from PIL import Image
from utils.data_pps import get_lists # 可能会有问题
import utils.config as config

    
# 传入 ['file_path',class] 返回img,class,path(用于看脏数据)
class MYDataset(Dataset):
    def __init__(self, paths, labels, transform):

        self.paths = paths
        self.labels = labels
        self.transform = transform
 
    def __len__(self):
        return len(self.paths)
 
    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx], self.paths[idx]

#采用prefetch_generator对DataLoader进行包装，可以提高数据读取速度        
class DataLoader_pre(torch.utils.data.DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class test_dataset_tta(Dataset):
    def __init__(self,paths,labels,tf_list=None):
        
        self.paths = paths
        self.labels = labels
        self.tf_list = tf_list
    
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self,idx):
        img_set = []
        image = Image.open(self.paths[idx]).convert('RGB')
        if self.transform is not None:
            for tf in tf_list:
                img_set.append(tf(image))
        return img_set, self.labels[idx]
        
        





def get_loader():    #  image folder 能够直接返回数据集，不用data_pps,但是不灵活
   
    data_transforms = {'train':config.train_transform,'val':config.test_transform}
    
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(config.root, x), data_transforms[x]) for x in ['train', 'val']}

    train_loader = torch.utils.data.DataLoader(
        image_datasets['train'],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        image_datasets['val'],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    # test_loader = torch.utils.data.DataLoader(
        # image_datasets['test'],
        # batch_size=batch_size,
        # num_workers=num_workers,
        # shuffle=False,
        # pin_memory=True,
        # drop_last=False,
    # )
    dataloaders_dict = {'train':train_loader,'val':val_loader}
    
    return dataloaders_dict , image_datasets['train'].class_to_idx
    

def get_debug_loader(type=1,merge=False,img_dir=''):    #  得到的dataloader 能够返回路径
    
    paths_all,labels,cls2id = get_lists(type=type,merge=merge,img_dir=img_dir)
    
    
    train_loader = DataLoader_pre(MYDataset(paths_all['train'],labels['train'],config.train_transform), 
                                            batch_size=config.batch_size, 
                                            shuffle=True, 
                                            pin_memory=True)
                                            
    val_loader = DataLoader_pre(MYDataset(paths_all['val'],labels['val'],config.test_transform), 
                                        batch_size=config.batch_size, 
                                        shuffle=False, 
                                        pin_memory=True)
    
    dataloaders_dict = {'train':train_loader,'val':val_loader}
    
    return dataloaders_dict,cls2id
    
    
def get_tta_loader():  # 对于val 做tta
    
    paths_all,labels,cls2id = get_lists(type=type,merge=merge,img_dir=img_dir)
    
    tta_loader = DataLoader_pre(test_dataset_tta(paths_all['val'],labels['val'],config.tta_trans_list), 
                                        batch_size=config.batch_size, 
                                        shuffle=True, 
                                        pin_memory=True)
    
    