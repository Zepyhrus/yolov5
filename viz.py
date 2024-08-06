import os
from os.path import join, basename
import cv2, numpy as np

from glob import glob
from urx.toolbox import sload
from urx.imgbox import rectangle
from urx.constants import COLORS

if __name__ == '__main__':
  tar = 'asher'
  seg = 'seg' in tar
  cfg = sload(f'data/{tar}.yaml')
  images = sorted(glob(f'data/{tar}/{cfg["train"]}/*.png'))
  assert len(images), 'No images found!'

  for image in images:
    print(image)
    img = cv2.imread(image)
    h, w, c = img.shape
    iid, *_ = os.path.splitext(basename(image))

    lfile = f'data/{tar}/labels/train/{iid}.txt'
    with open(lfile, 'r') as f:
      labels = f.readlines()

      for j, label in enumerate(labels):
        lbs = [_ for _ in label.split()]
        cls = int(lbs[0])
        # if cls != 4: continue

        color = COLORS[j % len(COLORS)]
        if seg:
          pts = np.array([float(_) for _ in lbs[1:]]).reshape((-1, 2))
          pts[:, 0] *= w
          pts[:, 1] *= h

          cv2.drawContours(img, [pts[:, None, :].astype(np.int32)], 0, color=color, thickness=2)
        else:
          cx, cy, tw, th = [float(_) for _ in lbs[1:]]
          lux = int((cx-tw/2) * w)
          luy = int((cy-th/2) * h)
          rbx = int((cx+tw/2) * w)
          rby = int((cy+th/2) * h)

          rectangle(img, (lux, luy, rbx, rby), color)

    cv2.imshow('_', img)
    if cv2.waitKey(0) == 27: break

