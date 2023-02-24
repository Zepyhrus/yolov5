import yaml, os
from os.path import join, basename
import cv2

from glob import glob


from utils.general import yaml_load

if __name__ == '__main__':
  tar = 'asher'
  cfg = yaml_load(f'data/{tar}.yaml')
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

      for label in labels:
        cls, cx, cy, tw, th = [float(_) for _ in label.split()]
        lux = int((cx-tw/2) * w)
        luy = int((cy-th/2) * h)
        rbx = int((cx+tw/2) * w)
        rby = int((cy+th/2) * h)

        cv2.rectangle(img, (lux, luy), (rbx, rby), (0, 255, 0))


    cv2.imshow('_', img)
    if cv2.waitKey(0) == 27: break

