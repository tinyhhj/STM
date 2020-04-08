import cv2
import matplotlib.pyplot as plt
import numpy as np
cap = cv2.VideoCapture('video-clip.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
print('fps',fps)
# prev = None
cnt =0
data = 'frames'
import os
os.makedirs(data,exist_ok=True)
# while cap.isOpened():
#     ret, frame = cap.read()
#     cv2.imwrite(f'{data}/{cnt:05d}.jpg',frame)
#     cnt+=1

image = cv2.imread(os.path.join(data,'car', '00000.jpg'))
# mask = cv2.imread('00000.png')
# print(mask.shapeqqqq)
print(image.shape)
orig_h, orig_w = image.shape[:-1]
# image = cv2.resize(image, (orig_w//3, orig_h//3))

# image
# masking window
class Sketcher(object):
    def __init__(self,img):
        h,w = img.shape[:-1]
        mask = np.zeros((h,w))
        self.prev_pt = None
        self.dests = [img,mask]
        self.show()
        cv2.setMouseCallback('image',self.onMouseClick)
    def show(self):
        cv2.imshow('image', self.dests[0])
        cv2.imshow('mask', self.dests[1])
    def onMouseClick(self,event, x, y, flags, params):
        pt = (x,y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # print('down')
            self.prev_pt = pt
        if event == cv2.EVENT_LBUTTONUP:
            # print('up')
            self.prev_pt = None
        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            # print('drag')
            cv2.line(self.dests[0],self.prev_pt, pt, (255,255,255),10)
            cv2.line(self.dests[1],self.prev_pt, pt , (1),10)
            self.prev_pt = pt
            self.show()

sketcher = Sketcher(image)


while True:
  key = cv2.waitKey()

  if key == ord('q'): # quit
    break
  if key == ord('s'): # reset
    # mask = cv2.resize(sketcher.dests[1],(orig_w,orig_h),interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('masks/car/00000.png', sketcher.dests[1])


