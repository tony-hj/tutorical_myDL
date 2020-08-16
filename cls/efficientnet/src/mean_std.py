# -*- coding: utf-8 -*-
import numpy as np
import cv2
import random
import os
from utils.data_pps import get_lists
import utils.config as config
from tqdm import tqdm

def calc_mean_std():
    path, _, _ = get_lists(config.root)
    path = path['train']+path['val']
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    print('default mean & std: \n', mean, std)
    means = [0, 0, 0]
    std = [0, 0, 0]

    index = 1
    num_imgs = 0
    for each in tqdm(path):
        num_imgs += 1
        img = cv2.imread(each).astype(np.float32) / 255.
        for i in range(3):
            means[i] += img[:, :, i].mean()
            stdevs[i] += img[:, :, i].std()

    means.reverse()
    std.reverse()

    means = np.asarray(means) / num_imgs
    std = np.asarray(stdevs) / num_imgs
    print('our dataset: \n', mean, std)
    return means,std



    

