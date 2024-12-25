from tqdm import tqdm
from glob import glob
from copy import deepcopy

import cv2, numpy as np, random, itertools as it
# from imgaug.augmentables import Keypoint, KeypointsOnImage
# from sympy import Point, Polygon 
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.collection import GeometryCollection


from urx.toolbox import sload, sdump
from urx.imgbox import rectangle, square_n_pad, iou



# 231031，将董令彩的两份圆杆数据集集中到一起来，并进行处理


labels = glob('.datasets/imgs1/*.json')
images = [_.replace('.json', '.bmp') for _ in labels]
random.shuffle(labels)

# # 初步查验数据
# for i, image in enumerate(tqdm(images)):
#   img = cv2.imread(image)

#   cv2.imshow('_', img)
  

#   k = cv2.waitKey(0)
#   if k == 27: break

def label2boxes():
  # 检测
  # 将所有的检测数据导入到asher
  start = 167
  for i, label in enumerate(tqdm(labels)):
    lb = sload(label)
    lb_new = deepcopy(lb)
    lb_new['shapes'] = []

    image = label.replace('.json', '.bmp')
    img = cv2.imread(image)
    img_new = img.copy()

    for shape in lb['shapes']:
      if shape['label'] != 'body': continue

      _ctr = np.array(shape['points'])
      _x,_y,_w,_h = cv2.boundingRect(_ctr.astype(np.float32))

      shape_new = deepcopy(shape)
      shape_new['label'] = 'tbar'
      shape_new['points'] = [[_x-_w*0.1, _y-_h*0.1], [_x+_w*1.1, _y+_h*1.1]]
      shape_new['shape_type'] = 'rectangle'

      lb_new['shapes'].append(shape_new)

      rectangle(img, [_x, _y, _x+_w, _y+_h])
    
    lb_new['imagePath'] = f'{str(start).zfill(8)}.png'
    lb_new['imageData'] = None

    cv2.imwrite(f'.datasets/soo/{lb_new["imagePath"]}', img_new)
    sdump(f'.datasets/soo/{str(start).zfill(8)}.json', lb_new)
    start += 1

    cv2.imshow('_', img)
    if cv2.waitKey(100) == 27: break
    if start >= 200: break


def label2masks():
  # 分割
  ious = []
  for i, label in enumerate(labels):
    lb = sload(label)
    img = cv2.imread(label.replace('.json', '.bmp'))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # 先把所有的contour画出来看看
    # for shape in lb['shapes']:
    #   _cotr = np.array(shape['points'], np.int32)

    #   img_blank = np.zeros_like(img_gray, dtype=np.uint8)
    #   cv2.drawContours(img_blank, [_cotr], 0, 255, cv2.FILLED)

    #   cv2.imshow('_', img_blank)
    #   k = cv2.waitKey(0)
    #   if k == 27: 
    #     exit()

    _bs = [_ for _ in lb['shapes'] if _['label'] == 'body']
    _as = [_ for _ in lb['shapes'] if _['label'] == 'upper']

    assert len(_bs)+len(_as) == len(lb['shapes']), "有其他label？"

    for _b in _bs:  # _b 一定是body
      _cb = np.array(_b['points'], np.int32)
      img_b = np.zeros_like(img_gray, dtype=np.uint8)
      cv2.drawContours(img_b, [_cb], 0, 1, cv2.FILLED)

      # 对每个body进行循环
      for _a in _as:  # _a 一定是upper
        _ca = np.array(_a['points'], np.int32)
        img_a = np.zeros_like(img_gray, dtype=np.uint8)
        cv2.drawContours(img_a, [_ca], 0, 1, cv2.FILLED)
        

        # and 是求交，or 是求并
        img_i = np.logical_and(img_a, img_b)
        iou = float(img_i.sum() * 1 / (img_a.sum() + img_b.sum()))
        ious.append(iou)
        
        # 只处理有问题的配对
        if iou > 1e-3:  
          img_b = np.logical_and(img_b, 1-img_a)

      # 针对所有的upper都取差完后可视化
      # cv2.imshow('_', img_b.astype(np.uint8)*255)
      # print(iou)
      # k = cv2.waitKey(200)
      # if k == 27:  exit()

      # 提取做差后的contour
      cotrs, *_ = cv2.findContours(img_b.astype(np.uint8)*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      areas = [cv2.contourArea(_) for _ in cotrs]
      cotr = cotrs[np.argmax(areas)]

      # 可视化
      # img_u = np.zeros_like(img_gray, dtype=np.uint8)
      # cv2.drawContours(img_u, [cotr], 0, 255, cv2.FILLED)
      # cv2.imshow('_', img_u)
      # k = cv2.waitKey(0)
      # if k == 27:  exit()
      
      _b['points'] = cotr[:, 0, :].tolist()
    lb['shapes'] = _bs + _as
    lb['imagePath'] = f'{i}.png'
    sdump(f'.datasets/soo/{i}.json', lb)
    cv2.imwrite(f'.datasets/soo/{i}.png', img)



if __name__ == '__main__':
  # 将soo数据集中的数据提取到256图像中去
  labels = glob('.datasets/soo/*.json')
  images = [_.replace('.json', '.png') for _ in labels]
  cnt = 438

  for label, image in zip(labels, images):
    img = cv2.imread(image)
    lb = sload(label)

    bodys = [_ for _ in lb['shapes'] if _['label'] == 'body']
    close_boxes = [
      cv2.boundingRect(np.array(_['points'], np.int32))
      for _ in bodys
    ]

    # 选取方框作为图像
    for x,y,w,h in close_boxes:
      img_s, box = square_n_pad((x,y,x+w,y+h), img, 0.2)
      img_b = cv2.resize(img_s, (256, 256))
      _img = img_b.copy()
      x1, y1, x2, y2 = box
      l = x2-x1
      
      new_lb = {
        'version': '5.2.1',
        'flags': {},
        'imagePath': f'{str(cnt).zfill(8)}.png',
        'shapes': [],
        'imageData': None,
      }
      for shape in lb['shapes']:
        kp = np.array(shape['points'], float)


        kp[:, 0] -= x1
        kp[:, 1] -= y1
        kp *= 256/l

        new_kp = Polygon(kp)
        new_box = Polygon([(0, 0), (256, 0), (256, 256), (0, 256)])
        color = (0, 255, 0) if shape['label'] == 'body' else (0, 0, 255)
        
        if new_kp.intersects(new_box):
          cotr = new_kp.intersection(new_box)
          if isinstance(cotr, (GeometryCollection, MultiPolygon)):
            areas = [_.area for _ in cotr.geoms]
            cotr = cotr.geoms[np.argmax(areas)]
          
          if cotr.area < 10: continue

          _cotr = np.array(cotr.exterior.coords)


          cv2.drawContours(_img, [_cotr.astype(np.int32)], 0, color, 2)

          new_lb['shapes'].append({
            'label': shape['label'],
            'group_id': None,
            'description': '',
            'shape_type': 'polygon',
            'flags': {},
            'points': _cotr.tolist()
          })
        else:
          print('empty')

      cv2.imshow('_', _img)
      if cv2.waitKey(1) == 27: exit()
      if cnt > 500: 
        cv2.imwrite(f'.datasets/tt/{str(cnt).zfill(8)}.png', img_b)
        sdump(f'.datasets/tt/{str(cnt).zfill(8)}.json', new_lb)
      else:
        cv2.imwrite(f'.datasets/tot/{str(cnt).zfill(8)}.png', img_b)
        sdump(f'.datasets/tot/{str(cnt).zfill(8)}.json', new_lb)
      cnt += 1


