# %% display video frame
import sys
sys.path.append('./yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
import os
import cv2

from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image

print('load module ok')

#%%load dataset
imgsz = 640
dataset = LoadImages('data/test.mp4', img_size=imgsz)

for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
    print( '\n' )
    print(f' frame index : {frame_idx} \n')
    display( Image.fromarray(cv2.cvtColor(im0s,cv2.COLOR_BGR2RGB)))
    if frame_idx >= 1 :
        break;

