import yaml
from os.path import join, basename
import cv2

from glob import glob


from utils.general import yaml_load

if __name__ == '__main__':
  tar = 'aug'
  coco = yaml_load(f'data/{tar}.yaml')

  images = sorted(glob(f'data/{tar}/images/train/*.png'))

  for image in images:
    img = cv2.imread(image)
    h, w, c = img.shape
    iid = basename(image)[:-4]

    lfile = f'data/{tar}/labels/train/{iid}.txt'
    with open(lfile, 'r') as f:
      labels = f.readlines()

      for label in labels:
        cls, cx, cy, tw, th = [float(_) for _ in label.split()]
        lux = int((cx-tw/2) * w)
        luy = int((cy-th/2) * h)
        rbx = int((cx+tw/2) * w)
        rby = int((cy+th/2) * h)

        cv2.rectangle(img, (lux, luy), (rbx, rby), (0, 255, 0) if cls else (255, 0, 0), 2)


    cv2.imshow('_', cv2.resize(img, (512, 512)))
    if cv2.waitKey(0) == 27: break

