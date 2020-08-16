import os
import numpy as np
from shutil import copyfile
from random import shuffle
from glob import glob
import pandas as pd

'''
file 文件名
paths 完整路径
label 数字
cls 单词
id 数字
'''

def get_lists(root):# mydict指定***.txt的地址
    
    '''
    returns
    1. path 一个字典，分为'train'、'val'，他们分别包含着由绝对路径组成的列表
    2. label ... 由数字标签组成
    3. cls2id {cls_1:0, cls_2:1}
    '''
    
    # 本次针对海洋生物数据集
    species = pd.read_csv(os.path.join(root,'species.csv'))
    cls2id = {list(species.ScientificName)[i]:list(species.ID)[i] for i in range(species.shape[0])}

    data = pd.read_csv(os.path.join(root,'training.csv'))
    data = data.sample(frac=1)
    paths2train = [root+'/data/'+i+'.jpg' for i in list(data['FileID'])]
    labels2train = list(data['SpeciesID'])

    anno = pd.read_csv(os.path.join(root,'annotation.csv')) # test.csv文件名和真实标签，用于验证
    paths4val = [root+'/data/'+i+'.jpg' for i in list(anno['FileID'])]
    labels4tval = list(anno['SpeciesID'])

    path = {}
    path['train'] = paths2train
    path['val'] = paths4val

    label = {}
    label['train'] = labels2train
    label['val'] = labels4tval

    return path,label,cls2id






