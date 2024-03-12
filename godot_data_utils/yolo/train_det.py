from ultralytics import YOLO

task='detection'
model = YOLO('yolov8n.yaml')
    
model.train(
    data='2024-shapes-det.yaml', 
    epochs=100, 
    save=True,
    workers=4,
    cos_lr=True,
    overlap_mask=False,
    # device=[0,1]
)