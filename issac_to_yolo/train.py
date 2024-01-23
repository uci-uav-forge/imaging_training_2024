from ultralytics import YOLO
from yolo_config import *

task='detection'
if task=='segmentation':
    model = YOLO('yolov8n-seg.yaml')
elif task=='detection':
    model = YOLO('yolov8n.yaml')
else:
    raise NotImplementedError
    
model.train(
    data='/home/eesh/forge/YOLO_DATASET/DATASETv1/data.yaml', 
    epochs=100,
    workers=16, 
    save=True,
    cos_lr=True
    )