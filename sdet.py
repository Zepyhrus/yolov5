# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
from glob import glob
from pathlib import Path
from tqdm import tqdm

import torch
import cv2

from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes)
from utils.torch_utils import select_device

if __name__ == "__main__":
    dir_data = 'data/asher'
    # images =  glob(f'{dir_data}/*.png')
    images = glob('/media/user/Sherk2T/Datasets/COCO/2017/train2017/*.jpg')
    print(len(images))

    # name = 'asher202303202'
    # weights = f'runs/train/{name}/weights/best.pt'

    weights = f'yolov5n.pt'
    nosave = False
    project = './runs/detect'
    exist_ok = True
    save_txt = True
    device = 'cpu'
    data = f'{dir_data}.yaml'
    dnn = False
    half = False
    imgsz = (640, 480)
    vid_stride = 1
    augment = False
    conf_thres = 0.5
    iou_thres = 0.4
    classes = None
    agnostic_nms = False
    save_crop = False
    save_conf = False
    line_thickness = 2
    hide_labels = False
    hide_conf = False
    max_det = 1000
    visualize = False
    view_img = False

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt    

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    
    img_ori = cv2.imread(images[0])
    im0s = cv2.resize(img_ori, imgsz)
    for _ in tqdm(images):
        im = im0s.transpose((2, 0, 1))

        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0s size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape)  # .round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    if c: continue

                    ulx, uly, brx, bry = [int(_) for _ in xyxy]
                    cx = int((xyxy[0] + xyxy[2]) / 2)
                    cy = int((xyxy[1] + xyxy[3]) / 2)

                    cv2.rectangle(im0s, (ulx, uly), (brx, bry), (0, 255, 0))


        # # Show results (image with detections)
        # cv2.imshow('_', im0s)
        # if cv2.waitKey(1) == 27: break

    