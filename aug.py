import shutil, os
from os.path import splitext, join, basename
from glob import glob
from tqdm import tqdm

import cv2, numpy as np


import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import json


def demo_aug():
  ia.seed(1)
  
  image = ia.quokka(size=(256, 256))
  bbs = BoundingBoxesOnImage([
    BoundingBox(x1=65, y1=100, x2=200, y2=150),
    BoundingBox(x1=150, y1=80, x2=200, y2=130)
  ], shape=image.shape)

  seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
    iaa.Affine(
      translate_px={"x": 40, "y": 60},
      scale=(0.5, 0.7)
    ) # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
  ])

  # Augment BBs and images.
  image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

  # print coordinates before/after augmentation (see below)
  # use .x1_int, .y_int, ... to get integer coordinates
  for i in range(len(bbs.bounding_boxes)):
    before = bbs.bounding_boxes[i]
    after = bbs_aug.bounding_boxes[i]
    print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
      i,
      before.x1, before.y1, before.x2, before.y2,
      after.x1, after.y1, after.x2, after.y2)
    )

  # image with BBs before/after augmentation (shown below)
  image_before = bbs.draw_on_image(image, size=2)
  image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])

  cv2.imshow('_', np.hstack((image_after, image_before)))
  cv2.waitKey(0)


def cleanup(prj='tarball'):
  shutil.rmtree(f'data/{prj}', True)
  os.makedirs(f'data/{prj}')

  labels = glob(f'backup/{prj}/*.json')
  images = glob(f'backup/{prj}/*.png')

  images_label = [_.replace('.json', '.png') for _ in labels]

  for i in range(len(images)):
    image = images[i]
    uid = str(i).zfill(8)
    dst_image = f'data/{prj}/{uid}.png'
    dst_label = f'data/{prj}/{uid}.json'

    shutil.copy(image, dst_image)

    if image in images_label:
      label = image.replace('.png', '.json')
      with open(label, 'r') as f:
        jabel = json.load(f)
        jabel['imagePath'] = f'{uid}.png'

      with open(dst_label, 'w') as f:
        json.dump(jabel, f, indent='\t')


def jload(filename):
  with open(filename, 'r') as f:
    return json.load(f)

if __name__ == '__main__':
  prj = 'asher'
  cleanup(prj)

  save = True
  start = len(glob(f'data/{prj}/*.png'))
  labels = glob(f'data/{prj}/*.json')

  seq = iaa.SomeOf(3, [
    iaa.Multiply((0.7, 1.5)), # change brightness, doesn't affect BBs
    iaa.Affine(
      translate_px={"x": (-40, 40), "y": (-40, 40)},
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

  for _ in tqdm(range(500)):
    label = np.random.choice(labels)
    image_label = label.replace('.json', '.png')
    lb = jload(label)
    img = cv2.imread(image_label)


    bbs = []
    for shape in lb['shapes']:
      x1, y1 = shape['points'][0]
      x2, y2 = shape['points'][1]
      bbs.append(BoundingBox(x1, y1, x2, y2, label=shape['label']))

    bbs = BoundingBoxesOnImage(bbs, shape=img.shape)

    img, bbs = seq(image=img, bounding_boxes=bbs)
    bbs = bbs.clip_out_of_image()

    if not len(bbs): continue
    
    new_idx = str(start+_).zfill(8)
    
    lb['imagePath'] = f'{new_idx}.png'
    lb['shapes'] = []
    for bb in bbs:
      shape = {
        'label': bb.label,
        'points': [
          [bb.x1, bb.y1],
          [bb.x2, bb.y2]
        ],
        'group_id': None,
        'shape_type': "rectangle",
        'flags': {},
      }
      lb['shapes'].append(shape)

    if save:
      cv2.imwrite(f'data/{prj}/{new_idx}.png', img)
      with open(f'data/{prj}/{new_idx}.json', 'w') as f:
        json.dump(lb, f, indent=4)

      # img_show = bbs.draw_on_image(img)
      # cv2.imshow('_', img_show)
      # k = cv2.waitKey(1)
      # if k == 27: break
