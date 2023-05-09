import argparse
import time
from pathlib import Path

import os
import sys

root_path = os.path.dirname(os.path.abspath(__file__))
root_path_1 = '/'.join(root_path.split('/')[:-1])
sys.path.append(root_path)
sys.path.append(root_path_1)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from IPython import embed

sys.path.append('/home/gongjt4/graspnegt_ws/src/graspnet_pkg/src/yolov7')
from yolov7.models.experimental import attempt_load

# from models.experimental import attempt_load

from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import warnings
warnings.filterwarnings("ignore")



# imgsz=1280 
# def detect_icra(source, weights, imgsz, device, conf_thres, iou_thres, classes, agnostic_nms, augment, show_res=True):
#     # Initialize
#     device = select_device(device)
#     if device != 'cpu':
#         device_type = 'cuda'
#     half = device_type != 'cpu'
    
#     # Load model
#     model = attempt_load(weights, map_location=device)  # load FP32 model
#     stride = int(model.stride.max())  # model stride
#     imgsz = check_img_size(imgsz, s=stride)  # check img_size

#     model = TracedModel(model, device, imgsz)

#     if half:
#         model.half()  # to FP16
    
#     # LoadImages
#     # im0s = cv2.imread(source)  # BGR
#     img0 = source

#     img = letterbox(img0, imgsz, stride=stride)[0]
#     # Convert
#     img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#     img = np.ascontiguousarray(img)

#     # Get names and colors
#     names = model.module.names if hasattr(model, 'module') else model.names
#     colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

#     # Run inference
#     if device_type != 'cpu':
#         model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

#     img = torch.from_numpy(img).to(device)
#     img = img.half() if half else img.float()  # uint8 to fp16/32
#     img /= 255.0  # 0 - 255 to 0.0 - 1.0
#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)

#     # Inference
#     t1 = time_synchronized()
#     with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
#         pred = model(img, augment=augment)[0]
#     t2 = time_synchronized()

#     # Apply NMS
#     pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)
#     t3 = time_synchronized()
    
#     if len(pred) == 0:
#         print('pred is None!')
#         return None

#     assert len(pred) == 1
#     det = pred[0]
    
#     need_idx = model.names.index('完整的西红柿') + 1
#     if need_idx not in det[:, -1]:
#         print('pred is None!')
#         return None
#     det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
    
#     det_valid = det[:, -1] == need_idx
#     det = det[det_valid, :]

#     if show_res:
#         for *xyxy, conf, cls in reversed(det):
#             label = f'{names[int(cls)]} {conf:.2f}'
#             plot_one_box(xyxy, img0, label='tomato', color=colors[int(cls)], line_thickness=1)
#             cv2.imwrite('tomato.png', img0)


#     print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
#     return det[:, :4]
     

def load_model(weights, imgsz, device):
    device = select_device(device)
    if device != 'cpu':
        device_type = 'cuda'
    # half = device_type != 'cpu'
    half = False
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    model = TracedModel(model, device, imgsz)

    # if half:
    #     model.half()  # to FP16
    
    model_info = {
        'model': model,
        'device': device,
        'device_type': 'device_type',
        'stride': stride,
        'imgsz': imgsz,
        'half': half
    }
    print('load done!')
    return model_info

def detect_icra_new(model_info, source, conf_thres, iou_thres, 
                    classes, agnostic_nms, augment, show_res=True):
    
    model       = model_info['model']
    device      = model_info['device']
    device_type = model_info['device_type']
    stride      = model_info['stride']
    imgsz       = model_info['imgsz']
    half        = model_info['half']
    # model, device, device_type, stride, imgsz, half = load_model(weights, imgsz, device)
    # LoadImages
    # im0s = cv2.imread(source)  # BGR
    img0 = source

    img = letterbox(img0, imgsz, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    print("names ==========================", names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device_type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=augment)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)
    t3 = time_synchronized()
    
    if len(pred) == 0:
        print('pred is None!')
        return None

    assert len(pred) == 1
    det = pred[0]

  

    # embed()
    # need_idx = model.names.index('完整的西红柿') + 1
    # if need_idx not in det[:, -1]:
    #     print('pred is None!')
    #     return None
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
    
    # det_valid = det[:, -1] == need_idx
    # det = det[det_valid, :]

    names_new = ['empty plate', 'full tomato', 'induction cooker', 'pan bottom', 'lying bread', 'knife handle', 'blender handle', 'pan handle', 'toaster hole', 'vertical bread', 'cut tomato', 'toaster', 'knife', 'blender', 'bean bowl', 'egg bowl', 'sauce bottle', 'seasoning bottle']

    # names_new = [
    #     'empty_plate',
    #     'compelete_tomato',  # 1
    #     'cooker',
    #     'surface_of_pan',
    #     'laying_bread',
    #     'knife_handle',
    #     'agitator_handle',
    #     'pan_handle',
    #     'empty_toaster',
    #     'cooked_bread',
    #     'half_tomatoes',
    #     'toaster',
    #     'knife',
    #     'agitator',
    #     'bean_bowl',
    #     'egg_bowl',
    #     'sauce_bottle',  # jian liao ping
    #     'condiment_bottle'
    # ]

    if show_res:
        for *xyxy, conf, cls in reversed(det):
            label = f'{names_new[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)
            cv2.imwrite('~/det_ws/src/det_pkg/scripts/tomato.png', img0)
            

    print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
    return det  # (n, 6)  6: x1,y1, x2,y2    conf  class,  n: number of box
     

if __name__ == '__main__':

    image_path = '~/det_ws/src/det_pkg/scripts/cv_color.png'
    img = cv2.imread(image_path)

    source       = img
    # weights      = '/home/gongjt4/graspnegt_ws/src/graspnet_pkg/src/yolov7/checkpoint/yolov7.pt'
    weights      = '~/det_ws/src/det_pkg/yolov7/yolov7.pt'
    imgsz        = 640
    device       = 'cpu'
    conf_thres   = 0.25
    iou_thres    = 0.45
    classes      = None
    agnostic_nms = False
    augment      = False

    model_info = load_model(weights, imgsz, device)

    detect_icra_new(model_info, source, conf_thres, iou_thres, 
                    classes, agnostic_nms, augment)


# if __name__ == "__main__":
#     image_path = '/DATA_EDS/yanzj/workspace/code/yolov7/inference/images/horses.jpg'
#     img = cv2.imread(image_path)
#     source       = img
#     weights      = 'checkpoint/yolov7-e6e.pt'
#     imgsz        = 640
#     device       = '7'
#     conf_thres   = 0.25
#     iou_thres    = 0.45
#     classes      = None
#     agnostic_nms = False
#     augment      = False
#     detect_icra(source, weights, imgsz, device, conf_thres, iou_thres, classes, agnostic_nms, augment)





# CUDA_VISIBLE_DEVICES=7 python detect.py --device 7 --weights checkpoint/yolov7-e6e.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
