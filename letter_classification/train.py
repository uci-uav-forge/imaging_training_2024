from ultralytics import YOLO

model = YOLO('yolov8n-cls.yaml')
    
model.train(
    data='../../godot-data-gen/letter_data', 
    epochs=100, 
    save=True,
    imgsz=128
    # device=[0,1]
)