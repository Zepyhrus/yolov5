import os, shutil
from glob import glob
from tqdm import tqdm

import cv2, numpy as np


import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from urx.toolbox import jload, jdump, yload


# 将所有图像收集到一起
def reorg():
  labels = glob('data/asher_b/*.json')
  import random
  random.seed(98)
  random.shuffle(labels)

  for i, label in enumerate(labels):
    lb = jload(label)
    image_source = f'data/asher_b/{lb["imagePath"]}'
    z = str(i).zfill(8)

    image_tar = f'data/asher/{z}.png'
    lable_tar = f'data/asher/{z}.json'

    lb['imagePath'] = z + '.png'

    shutil.copy(image_source, image_tar)
    jdump(lable_tar, lb)



def offline_augmentation(prj, aug_ratio=3, save=True):
  cfg = yload(f'data/{prj}.yaml')
  classes = {}
  for k, v in cfg['names'].items():
    classes[v] = k
  
  # 收集正样本
  images_p = glob(f'data/{prj}/*.png')
  images_n = glob(f'/media/ubuntu/Sherk2T/Datasets/COCO/2017/train2017/*.jpg')
  start = len(images_p)
  labels = glob(f'data/{prj}/*.json')
  augsize = aug_ratio * start # 按照样本数据的3倍进行增强
  ratio_bg = 0.3 # 背景增强
  ratio_split = 0.1 # train/val split

  seq = iaa.SomeOf(3, [
    iaa.Multiply((0.7, 1.5)), # change brightness, doesn't affect BBs
    iaa.Affine(
      translate_percent=0.2,
      scale=(0.7, 1.5),
      rotate=(-180, 180),
      shear=(-45, 45),
    ),
    iaa.Fliplr(0.25),
    iaa.Flipud(0.25),
    iaa.CoarseDropout(per_channel=True),
    iaa.ChannelShuffle(0.25),
    iaa.Sharpen(alpha=(0, 0.5)),
  ])

  # TODO: 添加样本均衡
  for i in tqdm(range(augsize)):
    label = np.random.choice(labels)
    image_label = label.replace('.json', '.png')
    lb = jload(label)
    img_ori = cv2.imread(image_label)


    bbs = []
    for shape in lb['shapes']:
      assert shape['shape_type'] == 'rectangle'

      x1, y1 = shape['points'][0]
      x2, y2 = shape['points'][1]
      if x1 > x2: x1, x2 = x2, x1
      if y1 > y2: y1, y2 = y2, y1
      
      bbs.append(BoundingBox(x1, y1, x2, y2, label=shape['label']))

    bbs = BoundingBoxesOnImage(bbs, shape=img_ori.shape)

    img, bbs = seq(image=img_ori, bounding_boxes=bbs)
    h, w, c = img.shape
    bbs = bbs.clip_out_of_image() # 注意如果bbs变换到图片外，会引起训练yolo warning

    if not len(bbs): continue

    # 随机添加背景
    r = np.random.random()
    if r < ratio_bg:
      img_n = cv2.imread(np.random.choice(images_n))
      img_n = cv2.resize(img_n, (w, h))

      img = r*img_n.astype(np.double) + (1-r)*img.astype(np.double)
      img = img.astype(np.uint8)
    
    if save:
      new_idx = 'a' + str(start + i).zfill(8) # a表示增强的图片

      lb_txt = []
      for bbox in bbs:
        lb_line = f'{classes[bbox.label]} {bbox.center_x/w} {bbox.center_y/h} {bbox.width/w} {bbox.height/h}\n'
        lb_txt.append(lb_line)
      
      tar = 'train' if np.random.rand() > ratio_split else 'val'
      tar_images_folder = f'data/{prj}/images/{tar}'
      tar_labels_folder = f'data/{prj}/labels/{tar}'

      if not os.path.exists(tar_images_folder):
        os.makedirs(tar_images_folder)
      if not os.path.exists(tar_labels_folder):
        os.makedirs(tar_labels_folder)

      cv2.imwrite(f'{tar_images_folder}/{new_idx}.png', img)
      with open(f'{tar_labels_folder}/{new_idx}.txt', 'w') as f:
        f.writelines(lb_txt)
    else:
      img_show = bbs.draw_on_image(img)
      cv2.imshow('_', img_show)
      k = cv2.waitKey(0)
      if k == 27: break





if __name__ == '__main__':
  prj = 'asher'
  offline_augmentation(prj, 25)

  







