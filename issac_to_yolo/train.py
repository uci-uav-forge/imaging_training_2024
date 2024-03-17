from ultralytics import YOLO
from yolo_config import *

task='detection'
if task=='segmentation':
    model = YOLO('yolov8n-seg.yaml')
elif task=='detection':
    model = YOLO('yolov8n.yaml')
else:
    raise NotImplementedError

if __name__ == "__main__":
    model.train(
        data='C:\\code\\imaging_training_2024\\YOLO_DATASET\\DATASETv1\\data.yaml', 
        epochs=200,
        workers=16, 
        save=True,
        cos_lr=True,
        imgsz=640,
        batch=-1
    )
