import os
import glob
from shutil import copy
root = 'train_data'
newdir = 'dataset'

id2cls = {0: u'其他垃圾_一次性快餐盒', 1: u'其他垃圾_污损塑料', 2: u'其他垃圾_烟蒂', 3: u'其他垃圾_牙签', 4: u'其他垃圾_破碎花盆及碟碗', 5: u'其他垃圾_竹筷', 6: u'厨余垃圾_剩饭剩菜', 7: u'厨余垃圾_大骨头', 8: u'厨余垃圾_水果果皮', 9: u'厨余垃圾_水果果肉', 10: u'厨余垃圾_茶叶渣', 11: u'厨余垃圾_菜叶菜根', 12: u'厨余垃圾_蛋壳', 13: u'厨余垃圾_鱼骨', 14: u'可回收物_充电宝', 15: u'可回收物_包', 16: u'可回收物_化妆品瓶', 17: u'可回收物_塑料玩具', 18: u'可回收物_塑料碗盆', 19: u'可回收物_塑料衣架', 20: u'可回收物_快递纸袋', 21: u'可回收物_插头电线', 22: u'可回收物_旧衣服', 23: u'可回收物_易拉罐', 24: u'可回收物_枕头', 25: u'可回收物_毛绒玩具', 26: u'可回收物_洗发水瓶', 27: u'可回收物_玻璃杯', 28: u'可回收物_皮鞋', 29: u'可回收物_砧板', 30: u'可回收物_纸板箱', 31: u'可回收物_调料瓶', 32: u'可回收物_酒瓶', 33: u'可回收物_金属食品罐', 34: u'可回收物_锅', 35: u'可回收物_食用油桶', 36: u'可回收物_饮料瓶', 37: u'有害垃圾_干电池', 38: u'有害垃圾_软膏', 39: u'有害垃圾_过期药物'}
if not os.path.exists('dataset'):
    os.mkdir('dataset')
for each in glob.glob(os.path.join(root,'*.txt')):
    each = each[:-4]
    print(each)
    with open(each+'.txt') as f:
        cls = id2cls[int(f.readline().strip('\n').split(', ')[1])]
        if not os.path.exists(os.path.join(newdir,cls)):
            os.mkdir(os.path.join(newdir,cls))
        copy(each+'.jpg',os.path.join(newdir,cls,each.split('\\')[-1]+'.jpg'))

