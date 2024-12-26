import numpy as np, cv2
import matplotlib
from urx.imgbox import blurate

videos = [
  '1 - Trim.mp4',
  '2.mp4',
  '5.avi',
  '6.avi',
  '7.avi',
  '8.1.avi',
  '8.2.avi',
  '9.avi',
]


total = 313
intv = 200
for video in videos:
  cap = cv2.VideoCapture(f'videos/{video}')

  cnt = 0
  save = 0
  while True:
    _, img = cap.read()

    if not _:
      if cnt == 0:
        print(video) # if read failed, output result
      break
    
    cnt += 1

    # if blurate(img) < 25: continue

    if cnt % intv == 0:
      cv2.imwrite(f'videos/faces2/{total}.jpg', img)
      print(total)
      total += 1
      save += 1
      if save >= 10: break




