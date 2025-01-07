import cv2, numpy as np

from urx.imgbox import rectangle, blurate
from urx.toolbox import sload


humans = sload('resources/videos/humans.yml')

cnt = 0
st = 369
valid = 0
for hm in humans:
  print(hm)

  for video in hm['videos']:

    cap = cv2.VideoCapture(f'resources/videos/{video}')

    while True:
      _, frame = cap.read()
      cnt += 1
      if not _: break

      if blurate(frame) < 30: continue

      if valid % 200 == 0:
        cv2.imwrite(f'data/faces_250106/{st}.jpg', frame)
        st += 1
      valid += 1
    print(cnt, '-', valid)


