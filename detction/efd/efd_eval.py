import torch
from torch.backends import cudnn
import os
from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

compound_coef = 4
force_input_size = None  
use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['一次性快餐盒', '书籍纸张', '充电宝', '剩饭剩菜', '包', '垃圾桶', '塑料器皿', '塑料玩具', '塑料衣架', '大骨头', '干电池', '快递纸袋', '插头电线', '旧衣服', '易拉罐', '枕头', '果皮果肉', '毛绒玩具', '污损塑料', '污损用纸', '洗护用品', '烟蒂', '牙签', '玻璃器皿', '砧板', '筷子', '纸盒纸箱', '花盆', '茶叶渣', '菜帮菜叶', '蛋壳', '调料瓶', '软膏', '过期药物', '酒瓶', '金属厨具', '金属器皿', '金属食品罐', '锅', '陶瓷器皿', '鞋', '食用油桶', '饮料瓶', '鱼骨']

threshold = 0.5
iou_threshold = 0.5


input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size




model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),

                             # replace this part with your project's anchor config
                             ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                             scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

model.load_state_dict(torch.load('/content/drive/My Drive/efd/dataset/efficientdet-d4_8_7976.pth'))
print('loaded!')


if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()


def return_res(id,model=model,use_cuda=use_cuda,use_float16=use_float16):
  img_path = os.path.join('datasets/dataset/val',os.listdir('datasets/dataset/val')[id])
  ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)
  
  model.requires_grad_(False)
  model.eval()
  
  if use_cuda:
      x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
  else:
      x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

  x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

  with torch.no_grad():
      features, regression, classification, anchors = model(x)

      regressBoxes = BBoxTransform()
      clipBoxes = ClipBoxes()

      out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

  out = invert_affine(framed_metas, out)
  
  out_boxes=[]
  out_scores=[]
  out_classes=[]
  
  for i in range(len(ori_imgs)):
      if len(out[i]['rois']) == 0:
          continue

      for j in range(len(out[i]['rois'])):
          (x1, y1, x2, y2) = out[i]['rois'][j].astype(np.int)
          cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
          obj = obj_list[out[i]['class_ids'][j]]
          score = float(out[i]['scores'][j])
          
          out_boxes.append([x1,y1,x2,y2])
          out_scores.append(score)
          out_classes.append(obj)
  
    return out_boxes,out_scores,out_classes
          


