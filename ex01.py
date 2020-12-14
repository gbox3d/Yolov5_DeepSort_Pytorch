# %%
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


print('load module ok')


#%% initialize deepsort
config_deepsort = 'deep_sort_pytorch/configs/deep_sort.yaml'
cfg = get_config()
cfg.merge_from_file(config_deepsort)

print(cfg)

# deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
#                     max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
#                     nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
#                     max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
#                     use_cuda=True)

# %%s
out = 'inference/output'
device = select_device('')

if os.path.exists(out):
    shutil.rmtree(out)  # delete output folder
os.makedirs(out)  # make new output folder

half = device.type != 'cpu'  # half precision only supported on CUDA

print(device)

# %%
