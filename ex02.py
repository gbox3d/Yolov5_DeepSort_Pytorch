# %% detection 
import sys
sys.path.append('./yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn



from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image


print('load module ok')


# %% Load model
weights = 'yolov5/weights/yolov5s.pt'

# with torch.no_grad():
    
device = select_device('')
half = device.type != 'cpu'  # half precision only supported on CUDA

print(device)

model = torch.load(weights, map_location=device)[
    'model'].float()  # load to FP32
model.to(device).eval()
if half:
    model.half()  # to FP16
names = model.module.names if hasattr(model, 'module') else model.names
print(names)


# %% 이미지 인식 
imgsz = 640
dataset = LoadImages('data/test.mp4', img_size=imgsz)

img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
# run once
_ = model(img.half() if half else img) if device.type != 'cpu' else None

conf_thres = 0.4
iou_thres = 0.5

# im0s는 원본이미지 (numpy 형식) 
# img 는 토치 형식 이미지 
for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
    print('\n')
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

        # Inference
    # t1 = time_synchronized()
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(
        pred, conf_thres, iou_thres, classes=[0], agnostic=False)
    # t2 = time_synchronized()

    p, s, im0 = path, '', im0s

    # [[det],[det],....]
    for i, det in enumerate(pred):
        if det is not None and len(det):
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()

            print(img.shape[2:])
            print( det[:, :4])
            for *xyxy, conf, cls in reversed(det):
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                print('%12s %.2f %.2f %.2f %.2f %.2f' % (names[int(cls)], conf,c1[0],c1[1],c2[0],c2[1]))
                
                cv2.rectangle(im0, c1,c2, (0,0,255), thickness=3, lineType=cv2.LINE_AA)
            
            display(Image.fromarray(cv2.cvtColor(im0,cv2.COLOR_BGR2RGB)))

            

#2개 프레임만 
    if frame_idx >= 1 :
        break;


# %%
