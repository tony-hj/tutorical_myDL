import os
import numpy as np
from shutil import copyfile
from random import shuffle
from glob import glob
from config import root

def get_lists(root=root,type=1,merge=False,img_dir=''):# mydict指定***.txt的地址
    '''
    type-0  理想型
    |dataset-|-train-|-classes-|-files
             |-val-|-classes-|-files
             
    type-1  需要merge
    |dataset-|-train-|-classes-|-files
             |-val-|-classes-|-files
             |-test-|-classes-|-files
             
    type-2  直接列表
    |dataset-|-classes-|-files

    type-3  
    |dataset-|-name.jpg,name.txt(华为云专用)
    
    type-4                  img_dir is needed !!!
    |dataset-|-img_all
             |-classes-|-***.txt
    '''
    test_paths = []
    train_paths = []
    val_paths = []
    all_path = []
    
    classes = os.listdir(os.path.join(root,'train'))
    cls2id = {name:i for i,name in enumerate(classes)}
    
    labels = {}
    paths = {'test':test_paths,'train':train_paths,'val':val_paths,'all':all_path}
    type2names = {0:['train','val'],1:['test','train','val'],2:['all']}
    
    if type in [1,0,2]:
        
        for name in type2names[merge]:
            pre = os.path.join(root,name,cls) #前缀
            paths[name] += [[path,cls2id[name]] for path in os.listdir(pre)]


    elif type == 4:

        pre = img_dir
        for cls in os.listdir(root):
            for txt in glob(os.path.join(pre,cls)+'/*.txt'):       
                for name in type2names[merge]:
                    if name in txt:
                        with open(txt) as f:
                            paths[name] = [[os.path.join(pre,line.strip('\n')),cls2id[cls]] for line in f.readlines()]
                            
    if merge:
        paths['train']+=paths['val']
    
    for name in type2names[merge]:
        labels[name] = [i[1]for i in paths[name]]
        
    for name in type2names[merge]:
        paths[name] = [i[0]for i in paths[name]]
    
    return paths,labels,cls2id








