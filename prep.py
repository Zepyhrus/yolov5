from os.path import join, basename, splitext
import os, shutil, copy
from glob import glob
from tqdm import tqdm

import cv2, numpy as np, random


import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage

from urx.toolbox import sload
from urx.imgbox import rectangle
from urx.colors import COLORS

AUGSEQ = iaa.SomeOf(3, [
  iaa.Multiply((0.75, 1.25)), # change brightness, doesn't affect BBs
  # iaa.Affine( # 对于目标检测，尽量少用affine
  #   translate_percent=0.1,
  #   scale=(0.75, 1.25),
  #   rotate=(-180, 180),
  #   shear=(-45, 45),
  # ),
  iaa.Fliplr(0.25),
  iaa.Flipud(0.25),
  iaa.CoarseDropout(per_channel=True),
  iaa.ChannelShuffle(0.25),
  iaa.Sharpen(alpha=(0, 0.5)),
  iaa.ElasticTransformation(alpha=15, sigma=3),
])


if __name__ == '__main__':
  base_dir = 'data'
  prj = 'asher'
  seg = 'seg' in prj
  aug_ratio = 20 if seg else 5
  save = True
  itp_num = 32  # 对圆的插值点数
  ratio_bg = 0.25

  cfg = sload(f'data/{prj}.yaml')
  classes = {}
  for k, v in cfg['names'].items():
    classes[v] = k

  images_n = glob(f'/media/user/Sherk2T/Datasets/COCO/2017/train2017/*.jpg')
  assert len(images_n), "No negative backgrounds found!"
  
  labels = glob(f'{base_dir}/{prj}/*.json')
  random.shuffle(labels)
  # imgs = [cv2.imread(_.replace('.json', '.png')) for _ in labels]
  
  steps = int(aug_ratio * len(labels))
  print(len(labels))
  for j in tqdm(range(steps)):
    # 准备数据
    label = random.choice(labels)
    # if not '0000040' in label: continue # 筛选想看的图片
    lb = sload(label)
    img = cv2.imread(label.replace('.json', '.png'))
    h, w, *_ = img.shape

    # 生成新的数据
    tar = 'train' if j < 0.9*aug_ratio*len(labels)  else 'val' # 10%作为验证集
    tar_images_folder = f'data/{prj}/images/{tar}'
    tar_labels_folder = f'data/{prj}/labels/{tar}'
    os.makedirs(tar_images_folder, exist_ok=True)
    os.makedirs(tar_labels_folder, exist_ok=True)


    # 保存新数据
    abname = 'a'+str(j).zfill(8)
    image_tar = join(tar_images_folder, abname+'.png')
    label_tar = join(tar_labels_folder, abname+'.txt')

    # 随机添加背景
    r = np.random.random()*0.3
    if np.random.random() < ratio_bg:
      img_n = cv2.imread(np.random.choice(images_n))
      img_n = cv2.resize(img_n, (w, h))

      img = r*img_n.astype(np.double) + (1-r)*img.astype(np.double)
      img = img.astype(np.uint8)

    # ------------------------------------------- Bounding box ----------------------------------
    bbs = []
    for shape in lb['shapes']:
      if shape['shape_type'] != 'rectangle': continue

      x1, y1 = shape['points'][0]
      x2, y2 = shape['points'][1]
      if x1 > x2: x1, x2 = x2, x1
      if y1 > y2: y1, y2 = y2, y1
      
      bbs.append(BoundingBox(x1, y1, x2, y2, label=shape['label']))
    bbs = BoundingBoxesOnImage(bbs, shape=img.shape)

    img_aug, tar_aug = AUGSEQ(image=img, bounding_boxes=bbs)
    tar_aug = tar_aug.clip_out_of_image() # 注意如果bbs变换到图片外，会引起训练yolo warning

    # 保存数据
    lns = []
    for bbox in tar_aug:
      ln = f'{classes[bbox.label]} {bbox.center_x/w} {bbox.center_y/h} {bbox.width/w} {bbox.height/h}' + '\n'
      lns.append(ln)

    if save:
      cv2.imwrite(image_tar, img_aug)
      with open(label_tar, 'w') as f: f.writelines(lns)
    else:
      img_show = tar_aug.draw_on_image(img_aug, size=4)

      cv2.imshow('_', img_show)
      if cv2.waitKey(0) == 27: break
  

  # =========================== vizualization at once! =================================
  if not save: exit() # 只有当数据保存时才会立刻可视化

  cfg = sload(f'data/{prj}.yaml')
  images = sorted(glob(f'data/{prj}/{cfg["train"]}/*.png'))
  assert len(images), 'No images found!'

  for image in images:
    print(image)
    img = cv2.imread(image)
    h, w, c = img.shape
    iid, *_ = os.path.splitext(basename(image))

    lfile = f'data/{prj}/labels/train/{iid}.txt'
    with open(lfile, 'r') as f:
      labels = f.readlines()

      for j, label in enumerate(labels):
        lbs = [_ for _ in label.split()]
        cls = int(lbs[0])
        color = COLORS[cls]
        
        cx, cy, tw, th = [float(_) for _ in lbs[1:]]
        lux = int((cx-tw/2) * w)
        luy = int((cy-th/2) * h)
        rbx = int((cx+tw/2) * w)
        rby = int((cy+th/2) * h)

        rectangle(img, (lux, luy, rbx, rby), color)
        cv2.putText(img, str(cls), (lux, luy), 0, 1.0, color, 2)


    cv2.imshow('_', img)
    if cv2.waitKey(0) == 27: break






    
  
  




      
  
  

  







