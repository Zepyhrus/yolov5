# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py weights yolov5s-seg.pt source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  screen                          # screenshot
                                                                  path/                           # directory
                                                                  list.txt                        # list of images
                                                                  list.streams                    # list of streams
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python segment/predict.py weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with dnn
                                          yolov5s-seg_openvino_model     # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode

    


if __name__ == "__main__":
  weights = 'runs/train-seg/face241230/weights/best.pt'
  source = 'data/faces/*.jpg'
  data = 'data/faces.yaml'
  imgsz = (256,256)
  conf_thres = 0.25
  iou_thres = 0.45
  max_det = 1000
  device = 'cpu'
  view_img = True
  save_txt = False
  save_conf = False
  save_crop = False
  nosave = True
  classes = 0
  agnostic_nms = False
  augment = False
  visualize = False
  update = False
  project = 'runs/predict-seg'
  name = 'exp'
  exist_ok = False
  line_thickness = 2
  hide_labels = False
  hide_conf = False
  half = False
  dnn = False
  vid_stride = 1
  retina_masks = False


  # Directories
  save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
  (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

  # Load model
  device = select_device(device)
  model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
  stride, names, pt = model.stride, model.names, model.pt
  imgsz = check_img_size(imgsz, s=stride)  # check image size

  # Dataloader
  bs = 1  # batch_size
  dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

  # Run inference
  model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
  for path, im, im0s, vid_cap, s in dataset:
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
      im = im[None]  # expand for batch dim

    # Inference
    pred, proto = model(im)[:2]

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
    print(pred)

    # Process predictions
    for i, det in enumerate(pred):  # per image
      im0 = im0s.copy()
      annotator = Annotator(im0, line_width=line_thickness, example=str(names))
      
      if len(det):
        masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
        

        # Mask plotting
        annotator.masks(
          masks,
          colors=[colors(x, True) for x in det[:, 5]],
          im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
          255 if retina_masks else im[i]
        )

      # Stream results
      im0 = annotator.result()
      cv2.imshow('_', im0)
      if cv2.waitKey(0) == 27:  exit()

