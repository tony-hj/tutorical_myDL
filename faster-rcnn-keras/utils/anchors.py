import numpy as np
import keras
import tensorflow as tf
from utils.config import Config
import matplotlib.pyplot as plt

config = Config()
# 生成不同大小的先验框
def generate_anchors(sizes=None, ratios=None):
    if sizes is None:
        sizes = config.anchor_box_scales

    if ratios is None:
        ratios = config.anchor_box_ratios

    num_anchors = len(sizes) * len(ratios)

    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T
    
    for i in range(len(ratios)):
        anchors[3*i:3*i+3, 2] = anchors[3*i:3*i+3, 2]*ratios[i][0]
        anchors[3*i:3*i+3, 3] = anchors[3*i:3*i+3, 3]*ratios[i][1]
    

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors
    # anchors will be like the following
    '''
    [[-64,-64,64,64],
     [-128,-128,128,128],
     [-64,-128,64,-128],
     [..,..,..,..]]
    '''


def shift(shape, anchors, stride=config.rpn_stride):
    # [0,1,2,3,4....37]->
    # [0.5,1.5,2.5...37.5]->
    # [8,24,....600]
    shift_x = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # 得到每个网格的中心
    # 每个网格是16*16
    # 最后一个中心是600,600：这个不影响结果
    #print(shift_x,shift_y)->[8. 24. 40.  ...   568. 584. 600.] [8. 8. 8. ... 600. 600. 600.]
    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])

    shifts = np.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)
    # 经过stack的变换之后可以和先验框直接进行相加--》得到基于每个网格的9*38*38个先验框
    shifts            = np.transpose(shifts)
    number_of_anchors = np.shape(anchors)[0]

    k = np.shape(shifts)[0]

    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
    return shifted_anchors# -》shape-(9*38*38,4)

def get_anchors(shape,width,height):
    # 用于获得先验框
    # shape--公用特征层的大小
    anchors = generate_anchors()
    network_anchors = shift(shape,anchors)
    network_anchors[:,0] = network_anchors[:,0]/width
    network_anchors[:,1] = network_anchors[:,1]/height
    network_anchors[:,2] = network_anchors[:,2]/width
    network_anchors[:,3] = network_anchors[:,3]/height
    network_anchors = np.clip(network_anchors,0,1)
    return network_anchors
