import argparse

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *

from hie.hie import HIE
from hie.hieval import HIEval
from hie.tools import jsdump

if __name__ == '__main__':
  gt = HIE('data/seed/labels/val.json', 'seed')
  anns = []  # result used to evaluate, containing only human
  project = 'yolov5'

  save_img        = True
  save_txt        = False
  source          = [gt._get_abs_name(_) for _ in gt.getImgIds()] # glob.glob('inference/images/*.jpg')
  weights         = 'weights/yolov5s.pt'
  view_img        = False
  webcam          = False
  out             = 'inference/output'
  imgsz           = 640
  augment         = True
  conf_thres      = 0.4
  iou_thres       = 0.5
  classes         = 0
  agnostic_nms    = True


  for conf_thres in [.02, .05, .1, .2, .3, .4, .5]:
    for iou_thres in [.1, .2, .3, .4, .5, .6, .7]:
      with torch.no_grad():
        # Initialize
        device = torch_utils.select_device('')
        if os.path.exists(out):
          shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
          model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
          modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
          modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
          modelc.to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
          view_img = True
          cudnn.benchmark = True  # set True to speed up constant image size inference
          dataset = LoadStreams(source, img_size=imgsz)
        else:
          save_img = True
          dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in tqdm(dataset, dynamic_ncols=True):
          img = torch.from_numpy(img).to(device)
          img = img.half() if half else img.float()  # uint8 to fp16/32
          img /= 255.0  # 0 - 255 to 0.0 - 1.0
          if img.ndimension() == 3:
            img = img.unsqueeze(0)

          # Inference
          t1 = torch_utils.time_synchronized()
          pred = model(img, augment=augment)[0]

          # Apply NMS
          pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
          t2 = torch_utils.time_synchronized()

          # Apply Classifier
          if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

          # Process detections
          for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
              p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
              p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
              # Rescale boxes from img_size to im0 size
              det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

              # Print results
              for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results to HIE format
                for *xyxy, conf, cls in det:
                  if int(cls) == 0:
                    x1, y1, x2, y2 = [int(_) for _ in xyxy]

                    ann = {
                      'bbox': [x1, y1, x2-x1, y2-y1],
                      'category_id': 0,
                      'score': float(conf),
                      'image_id': os.path.basename(path)[:-4]
                    }

                    anns.append(ann)

        dt = gt.load_res(anns)

        _eval = HIEval(gt, dt, 'bbox')
        msg, _ = _eval.new_summ()

        with open(f'det/{project}-iou-{iou_thres}-thresh-{conf_thres}.txt', 'a') as f:
          f.write(f'iou-{iou_thres}-thresh-{conf_thres}: {msg}\n')
