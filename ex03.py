# %% basic tracking
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


# 중심점과 높이 넓이 형식으로 변경
def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


print('load module ok')

# %% Load model
weights = 'yolov5/weights/yolov5s.pt'

# with torch.no_grad():

device = select_device('')
half = device.type != 'cpu'  # half precision only supported on CUDA

print(device)

model = torch.load(weights,
                   map_location=device)['model'].float()  # load to FP32
model.to(device).eval()
if half:
    model.half()  # to FP16
names = model.module.names if hasattr(model, 'module') else model.names
print(names)

#%% initialize deepsort
deep_sort_config_file = 'deep_sort_pytorch/configs/deep_sort.yaml'
cfg = get_config()
cfg.merge_from_file(deep_sort_config_file)

deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE,
                    n_init=cfg.DEEPSORT.N_INIT,
                    nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

# %% 이미지 인식
imgsz = 640
dataset = LoadImages('data/test.mp4', img_size=imgsz)  #로딩후 이미지크기를 조절

img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
# run once
_ = model(img.half() if half else img) if device.type != 'cpu' else None

conf_thres = 0.4
iou_thres = 0.5

# im0s는 원본이미지 (numpy 형식)
# img 는 토치 형식 이미지 (크기조절됨)
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
    pred = non_max_suppression(pred,
                               conf_thres,
                               iou_thres,
                               classes=[0],
                               agnostic=False)
    # t2 = time_synchronized()

    p, s, im0 = path, '', im0s
    

    # [[det],[det],....]
    for i, det in enumerate(pred):
        if det is not None and len(det):
            
            #(0~4) 번째 있는 바운딩 박스 정보를 원본 이미지에 맞게 스케일링
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                      im0.shape).round()

            bbox_xywh = []
            confs = []

            # Adapt detections to deep sort input format
            for *xyxy, conf, cls in det:
                x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                obj = [x_c, y_c, bbox_w, bbox_h]
                bbox_xywh.append(obj)
                confs.append([conf.item()])
            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)

            # 검출된 바운딩박스 정보를 딥소트에 넘겨주기 
            outputs = deepsort.update(xywhs, confss, im0)

            # draw boxes for visualization
            if len(outputs) > 0:
                result_img = im0.copy()
                for i, (*xyxy,_id) in enumerate(outputs) :
                    
                    c1 = (xyxy[0],xyxy[1])
                    c2 = (xyxy[2],xyxy[3])
                    cv2.rectangle(result_img, c1, c2, (0,0,255), thickness=3, lineType=cv2.LINE_AA)
                    label = f'id:{_id}'
                    cv2.putText(result_img, label, c1, cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

                # 선택된 아이디만 골라내기 
                # _obj = [_v for _v in outputs if _v[-1] == 1]
                # print(_obj[0])
                    
                    # draw_boxes(im0, bbox_xyxy, identities)
                display(Image.fromarray(cv2.cvtColor(result_img,cv2.COLOR_BGR2RGB)))

        else:
            deepsort.increment_ages()  # next step

#2개 프레임만
    if frame_idx >= 2:
        break

# %%
