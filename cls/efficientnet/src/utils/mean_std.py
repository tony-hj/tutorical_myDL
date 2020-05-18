# -*- coding: utf-8 -*-
import numpy as np
import cv2
import random
import os

# calculate means and std  注意换行\n符号
# train.txt中每一行是图像的位置信息
def calc_mean_std(path = 'train.txt')

    means = [0, 0, 0]
    stdevs = [0, 0, 0]

    index = 1
    num_imgs = 0
    with open(path, 'r') as f:
        lines = f.readlines()
        # random.shuffle(lines)

        for line in lines:
            index += 1
            a = os.path.join(line)
            # print(a[:-1])
            num_imgs += 1
            img = cv2.imread(a[:-1])
            img = np.asarray(img)

            img = img.astype(np.float32) / 255.
            for i in range(3):
                means[i] += img[:, :, i].mean()
                stdevs[i] += img[:, :, i].std()

    means.reverse()
    stdevs.reverse()

    means = np.asarray(means) / num_imgs
    stdevs = np.asarray(stdevs) / num_imgs
    return means,stdevs



