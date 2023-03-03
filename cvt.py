from tqdm import tqdm

import os, cv2, json, shutil, yaml, uuid
import numpy as np, random
from glob import glob

from os.path import basename, splitext


def cvt_labelme2yolov5(dir_data):
  # 将labelme数据级转化为coco数据集
  dir_labels = f'{dir_data}/labels/train'
  dir_images = f'{dir_data}/images/train'
  # images = glob(f'{dir_data}/*.png')
  labels = glob(f'{dir_data}/*.json')
  names = [
    'tarball',
    'chessboard',
    'arucoboard',
    'nbwl',
  ]

  # 创建数据文件夹
  shutil.rmtree(f'{dir_data}/images', ignore_errors=True)
  shutil.rmtree(f'{dir_data}/labels', ignore_errors=True)

  os.makedirs(dir_labels, exist_ok=True)  # labels/...
  os.makedirs(dir_images, exist_ok=True)  # images/...

  for idx in ['train', 'test', 'val']:
    os.makedirs(f'{dir_data}/images/{idx}', exist_ok=True)
    os.makedirs(f'{dir_data}/labels/{idx}', exist_ok=True)

  # 将labelme格式数据转化为COCO格式
  for label in tqdm(labels): # json file
    # 获取图片的iid
    iid, _ext = splitext(basename(label))
    image = f'{dir_data}/{iid}.png'
    img = cv2.imread(image)

    h, w, c = img.shape

    # 读取labelme格式标签
    with open(label, 'r') as f:
      pts = json.load(f)
    
    # 转换为yolo格式数据并存为txt
    with open(f'{dir_labels}/{iid}.txt', 'w') as f:
      for pt in pts['shapes']:
        ulx, uly = pt['points'][0]
        brx, bry = pt['points'][1]
        cx, cy, bw, bh = (ulx+brx)/2, (uly+bry)/2, np.abs(ulx-brx), np.abs(uly-bry)

        class_num = names.index(pt['label'])
        f.write(f'{class_num} {cx/w} {cy/h} {bw/w} {bh/h}\n')

    # 复制图片
    shutil.copy(image, f'{dir_images}/{iid}.png')

  # 保存项目配置文件
  with open(f'{dir_data}.yaml', 'w') as f:
    yaml.safe_dump({
      'path': dir_data,
      'train': 'images/train',
      'val': 'images/val',
      'names': {_: names[_] for _ in range(len(names))},
    }, f, sort_keys=False)


# 选择一块大小适中的棋盘格
def select_chessboard(label):
  cn, cx, cy, bw, bh = label[0]
  bn, bx, by, bbw, bbh = label[-1]

  assert cn == 0 and bn == 1, "Must we have a corner and a board!"

  # 选择一块棋盘格
  tried = 0
  while True:
    # random ratio of ulx, uly, w, h
    rrulx, rruly = np.random.uniform(0, 1-3*bw), np.random.uniform(0, 1-3*bh)
    rrw, rrh = np.random.uniform(3*bw, 1-rrulx), np.random.uniform(3*bh, 1-rruly)

    # 选择该棋盘格的标签
    lthan = np.logical_and(label[:, 1] > rrulx, label[:, 2] > rruly)
    sthan = np.logical_and(label[:, 1] <= rrulx+rrw, label[:, 2] <= rruly+rrh)
    
    # 先选角点
    rcorner = label[np.logical_and(np.logical_and(lthan, sthan), label[:, 0]==0)]
    if len(rcorner): break
    
    tried += 1
    if tried >= 10:
      raise Exception('Ill label! No corners for augumentation...')

  # 对结果进行裁减
  rcorner_ulx = np.clip(rcorner[:, 1] - rcorner[:, 3]/2, rrulx, rrulx+rrw)
  rcorner_uly = np.clip(rcorner[:, 2] - rcorner[:, 4]/2, rruly, rruly+rrh)
  rcorner_brx = np.clip(rcorner[:, 1] + rcorner[:, 3]/2, rrulx, rrulx+rrw)
  rcorner_bry = np.clip(rcorner[:, 2] + rcorner[:, 4]/2, rruly, rruly+rrh)

  rcorner[:, 1] = (rcorner_ulx + rcorner_brx) / 2
  rcorner[:, 2] = (rcorner_uly + rcorner_bry) / 2
  rcorner[:, 3] = rcorner_brx - rcorner_ulx
  rcorner[:, 4] = rcorner_bry - rcorner_uly

  # 再选整个棋盘格
  rboard_ulx, rboard_uly = max(bx-bbw/2, rrulx), max(by-bbh/2, rruly)
  rboard_brx, rboard_bry = min(bx+bbw/2, rrulx+rrw), min(by+bbh/2, rruly+rrh)
  rboard = np.array((1, (rboard_ulx+rboard_brx)/2, (rboard_uly+rboard_bry)/2, rboard_brx-rboard_ulx, rboard_bry-rboard_uly), )

  return np.vstack((rcorner, rboard)), rrulx, rruly, rrw, rrh


# 随机通道漂移增强
def rand_chan_shift(img):
  assert img.ndim == 3, 'Must be multiple channel image!'

  return np.ascontiguousarray(img[:, :, np.random.permutation(3)])

# 以一张正样本图像为背景、若干副样本生成图片
def gen_p_bg(dir_data, image_p, image_n):
  b = 0.5
  a_xyxy2ccwh = np.array([
    [   1,  0, -b,  0],
    [   0,  1,  0, -b],
    [   1,  0,  b,  0],
    [   0,  1,  0,  b],
  ])

  img_bg = cv2.imread(image_p)
  h, w, _ = img_bg.shape

  iid = os.path.basename(image_p)[:-4]
  lb = np.loadtxt(f'{dir_data}/labels/train/{iid}.txt', ndmin=2)
  lb_xyxy = lb.copy()
  lb_xyxy[:, 1:] = (a_xyxy2ccwh @ lb[:, 1:].T).T

  x1 = np.random.uniform(0, lb_xyxy[:, 1].min())
  y1 = np.random.uniform(0, lb_xyxy[:, 2].min())
  x2 = np.random.uniform(lb_xyxy[:, 3].max(), 1)
  y2 = np.random.uniform(lb_xyxy[:, 4].max(), 1)

  lb_ = lb.copy()
  lb_[:, 1] = (lb[:, 1] - x1) / (x2 - x1)
  lb_[:, 2] = (lb[:, 2] - y1) / (y2 - y1)
  lb_[:, 3] = lb[:, 3] / (x2 - x1)
  lb_[:, 4] = lb[:, 4] / (y2 - y1)

  if np.any(lb_[:, 1:] > 1):
    raise Exception(str(lb_))

  img_bg = img_bg[int(y1*h):int(y2*h), int(x1*w):int(x2*w), :]
  h, w, c = img_bg.shape

  img_fg = cv2.resize(cv2.imread(image_n), (w, h))
  gamma = np.random.uniform(0.5, 1)
  img = img_bg*gamma + img_fg*(1-gamma)
  img = img.astype(np.uint8)

  return img, lb_

# 以一张负样本图像为背景、若干棋盘格正样本生成图片
def gen_n_bg(image_n, images_p):
  img_n = rand_chan_shift(cv2.imread(image_n))
  h_n, w_n, c_n = img_n.shape

  rlabels = []
  for image_p in images_p:
    # 生成随机窗口
    rrulx_w, rruly_w = np.random.uniform(0, 1-1/np.sqrt(w_n)), np.random.uniform(0, 1-1/np.sqrt(h_n))
    rrbrx_w, rrbry_w = np.random.uniform(rrulx_w+1/np.sqrt(w_n), 1), np.random.uniform(rruly_w+1/np.sqrt(h_n), 1)

    # 生成随机正样本
    img_p = cv2.imread(image_p)
    h, w, c = img_p.shape

    iid = os.path.basename(image_p)[:-4]
    label_p = np.loadtxt(f'data/labels/train/{iid}.txt')
    
    # 生成随机网格标签
    rlabel, rrulx, rruly, rrw, rrh = select_chessboard(label_p)
    rulx, ruly = int(rrulx*(w)), int(rruly*(h))
    rw, rh = int(rrw*w), int(rrh*h)
    rimg_p = rand_chan_shift(img_p[ruly:ruly+rh, rulx:rulx+rw, ...])

    # 覆盖窗口中的数据
    rulx_w, ruly_w, rbrx_w, rbry_w = int(rrulx_w*w_n), int(rruly_w*h_n), int(rrbrx_w*w_n), int(rrbry_w*h_n)
    img_pw = cv2.resize(rimg_p, (rbrx_w-rulx_w, rbry_w-ruly_w))
    # 随机对图像更新权重
    gamma = np.random.uniform(0.5, 1)
    patch_n = img_n[ruly_w:rbry_w, rulx_w:rbrx_w, :].astype(np.float32)
    patch_p = img_pw.astype(np.float32)
    
    img_n[ruly_w:rbry_w, rulx_w:rbrx_w, :] = (patch_p*gamma + patch_n*(1-gamma)).astype(np.uint8)

    # 将随机标签从正样本坐标系变换到局部坐标系
    rlabel_local = rlabel.copy()
    rlabel_local[:, 1] = (rlabel_local[:, 1] - rrulx) / rrw
    rlabel_local[:, 2] = (rlabel_local[:, 2] - rruly) / rrh
    rlabel_local[:, 3] = rlabel_local[:, 3] / rrw
    rlabel_local[:, 4] = rlabel_local[:, 4] / rrh

    # 将随机标签从局部坐标系变换到负样本坐标系
    rlabel_n = rlabel_local.copy()
    rlabel_n[:, 1] = rlabel_local[:, 1] * (rrbrx_w - rrulx_w) + rrulx_w
    rlabel_n[:, 2] = rlabel_local[:, 2] * (rrbry_w - rruly_w) + rruly_w
    rlabel_n[:, 3] = rlabel_local[:, 3] * (rrbrx_w - rrulx_w)
    rlabel_n[:, 4] = rlabel_local[:, 4] * (rrbry_w - rruly_w)

    rlabels.append(rlabel_n)
  rlabels = np.vstack(rlabels)

  return img_n, rlabels

if __name__ == '__main__':
  dir_data = 'data/asher'

  cvt_labelme2yolov5(dir_data)
  # exit()
  
  images_p = glob(f'{dir_data}/images/train/*.png')
  images_n = glob(f'/media/ubuntu/Sherk2T/Datasets/COCO/2017/train2017/*.jpg')

  assert len(images_n) and len(images_p), "No images available"
  
  for _ in tqdm(range(500)):
    # 选择图片，生成随机背景
    image_n = random.choice(images_n)
    image_p = random.choice(images_p)
    # ps = random.sample(images_p, random.randint(1, 3))

    img, label = gen_p_bg(dir_data, image_p, image_n)
    h, w, c = img.shape

    # 保存标签
    uid = uuid.uuid4().hex
    tar = 'train' if np.random.random() > .1 else 'val'
    cv2.imwrite(f'{dir_data}/images/{tar}/{uid}.png', img)
    with open(f'{dir_data}/labels/{tar}/{uid}.txt', 'w') as f:
      for rc, rcx, rcy, rbw, rbh in label:
        f.write(f'{int(rc)} {rcx:.6f} {rcy:.6f} {rbw:.6f} {rbh:.6f}\n')
    
    # # 可视化以检查
    # for rc, rcx, rcy, rbw, rbh in label:
    #   cv2.rectangle(img, (int((rcx-rbw/2)*w), int((rcy-rbh/2)*h)), (int((rcx+rbw/2)*w), int((rcy+rbh/2)*h)), (255, 0, 0) if rc else (0, 255, 255), 1)
    # cv2.imshow('_', cv2.resize(img, (320, 320)))
    # if cv2.waitKey(1) == 27: break

