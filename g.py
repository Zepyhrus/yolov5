from tqdm import tqdm
from glob import glob
import cv2

from urx.toolbox import sload, sdump
from urx.imgbox import rectangle, square_n_pad



# 231031，将董令彩的两份圆杆数据集集中到一起来，并进行处理


labels = glob('.datasets/raw_labelled_ds/imgs*/*.json')
images = [_.replace('.json', '.bmp') for _ in labels]



for i, image in enumerate(tqdm(images)):
  img = cv2.imread(image)

  cv2.imshow('_', img)
  

  k = cv2.waitKey(0)
  if k == 27: break
