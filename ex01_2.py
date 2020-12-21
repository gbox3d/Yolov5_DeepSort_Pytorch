# %% display webcam video frame
import sys
sys.path.append('./yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
import os
import cv2
from pathlib import Path

from IPython.display import display
import PIL.Image as Image

print('load module ok')

#%%load dataset
imgsz = 640
# dataset = LoadImages('data/test.mp4', img_size=imgsz)
# dataset = LoadStreams('0', img_size=imgsz)
dataset = LoadStreams('rtsp://admin:28608010@ubiqos001.iptimecam.com:15102/stream_ch00_1', img_size=imgsz)

for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):

    im0 = im0s[0].copy() # webcam 일경우 배열형식이다,

    cv2.imshow('frame',im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print( f'\n {frame_idx}' )
    
