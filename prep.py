from os.path import join, basename, splitext
import os, shutil, copy
from glob import glob
from tqdm import tqdm

import cv2, numpy as np


import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage

from urx.toolbox import jload, jdump, yload

AUGSEQ = iaa.SomeOf(3, [
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
  iaa.ElasticTransformation(alpha=15, sigma=3),
])


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



if __name__ == '__main__':
  prj = 'asher'
  seg = 'seg' in prj
  aug_ratio = 50
  save = True
  itp_num = 16  # 插值点数
  ratio_bg = 0.3

  cfg = yload(f'data/{prj}.yaml')
  classes = {}
  for k, v in cfg['names'].items():
    classes[v] = k

  images_n = glob(f'/media/ubuntu/Sherk2T/Datasets/COCO/2017/train2017/*.jpg')
  assert len(images_n), "No negative backgrounds found!"
  labels = glob(f'data/{prj}/*.json')
  imgs = [cv2.imread(_.replace('.json', '.png')) for _ in labels]
  
  
  print(len(labels))
  for j in tqdm(range(aug_ratio * len(labels))):
    # 准备数据
    label = labels[j % len(labels)]
    lb = jload(label)
    img = imgs[j % len(labels)]
    h, w, *_ = img.shape

    # 生成新的数据
    tar = 'train' if np.random.random() > 0.1  else 'val'
    tar_images_folder = f'data/{prj}/images/{tar}'
    tar_labels_folder = f'data/{prj}/labels/{tar}'
    os.makedirs(tar_images_folder, exist_ok=True)
    os.makedirs(tar_labels_folder, exist_ok=True)


    # 保存新数据
    abname = 'a'+str(j).zfill(8)
    image_tar = join(tar_images_folder, abname+'.png')
    label_tar = join(tar_labels_folder, abname+'.txt')

    # 随机添加背景
    r = np.random.random()*0.7
    if np.random.random() < ratio_bg:
      img_n = cv2.imread(np.random.choice(images_n))
      img_n = cv2.resize(img_n, (w, h))

      img = r*img_n.astype(np.double) + (1-r)*img.astype(np.double)
      img = img.astype(np.uint8)

    if seg:
      # ----------------------------- Seg ----------------------------------------
      assert len(lb['shapes']) == 1, "#Labels must be less than 2 for seg!"

      kps_all = []
      kps = []
      for shape in lb['shapes']:
        x1, y1 = shape['points'][0]
        x2, y2 = shape['points'][1]

        a, b = (x2-x1)/2, (y2-y1)/2
        xc, yc = (x2+x1)/2, (y2+y1)/2
        theta = np.arange(itp_num) * 2*np.pi / itp_num

        xs = (np.cos(theta)*a + xc)
        ys = (np.sin(theta)*b + yc)
        
        for x, y in zip(xs, ys):
          kps.append(Keypoint(x, y))
        kps_all.append({'cls': shape['label'], 'len': itp_num})
      kps = KeypointsOnImage(kps, shape=img.shape)
      kps.clip_out_of_image()

      # 数据增强
      img_aug, tar_aug = AUGSEQ(image=img, keypoints=kps)
      tar_aug = tar_aug.clip_out_of_image() # 注意如果bbs变换到图片外，会引起训练yolo warning

      # 保存数据
      st = 0
      lns = []
      for i in range(len(kps_all)):
        cls = classes[kps_all[i]['cls']]
        kp = tar_aug[st:st+kps_all[i]['len']]

        cords = np.array([[_.x, _.y] for _ in kp])
        cords[:, 0] /= w
        cords[:, 1] /= h

        ln = ' '.join([str(_) for _ in [cls]+cords.flatten().tolist()]) + '\n'
        lns.append(ln)

        st += kps_all[i]['len']
    else:
      # ------------------------------------------- Bounding box ----------------------------------
      bbs = []
      for shape in lb['shapes']:
        assert shape['shape_type'] == 'rectangle'

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






    
  
  




      
  
  

  







