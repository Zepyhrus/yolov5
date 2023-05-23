from tqdm import tqdm
from glob import glob
import cv2

from urx.toolbox import sload, sdump

labels = glob('data/tarball-seg256/*.json')
images = glob('data/tarball-seg256/*.json')

for label in tqdm(labels):
  # 处理图片
  image = label.replace('.json', '.png')
  img = cv2.imread(image)
  img = cv2.resize(img, (256, 256))
  cv2.imwrite(image, img)

  # 处理标签
  lb = sload(label)
  for i in range(len(lb['shapes'])):
    for j in range(len(lb['shapes'][i]['points'])):
      lb['shapes'][i]['points'][j][0] *= 2
      lb['shapes'][i]['points'][j][1] *= 2
  sdump(label, lb)