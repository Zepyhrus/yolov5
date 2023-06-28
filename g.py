from tqdm import tqdm
from glob import glob
import cv2

from urx.toolbox import sload, sdump
from urx.imgbox import rectangle, square_n_pad

start0 = 155
start = 409



while True:
  lb = sload(f'data/asher/00000{start0}.json')
  img = cv2.imread(f'data/asher/00000{start0}.png')


  for shape in lb['shapes']:
    if shape['label'] != 'tarball': pass
    box = shape['points'][0] + shape['points'][1]

    img_sq, box_sq = square_n_pad(box, img)
    img_sq = cv2.resize(img_sq, (128, 128))

    cv2.imwrite(f'data/tarball-seg/00000{start}.png', img_sq)
    start += 1
    # if cv2.waitKey(0) == 27:
    #   raise Exception('Break manually...')
    

  start0 += 1
  


