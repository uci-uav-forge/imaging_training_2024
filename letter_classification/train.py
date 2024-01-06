from ultralytics import YOLO

model = YOLO('yolov8n-cls.yaml')
    
model.train(
    data='datasets/dataset', 
    epochs=30, 
    save=True,
    imgsz=128
    # device=[0,1]
)