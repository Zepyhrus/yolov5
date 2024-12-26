import numpy as np, cv2

from glob import glob
from urx.toolbox import sload, sdump

from tqdm import tqdm

for i in tqdm(range(313)):
  label = f'data/faces3/{i}.json'
  image = f'data/faces3/{i}.png'


  img = cv2.imread(image)
  lb = sload(label)
  lb['imagePath'] = f'{i}.jpg'


  sdump(f'data/faces/{i}.json', lb)
  cv2.imwrite(f'data/faces/{i}.jpg', img)
