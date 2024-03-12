from ultralytics import YOLO

# ultralytics==8.1.5
task='segmentation'
model = YOLO('yolov8n-seg.yaml')
    
model.train(
    data='2024-shapes-seg.yaml', 
    epochs=100, 
    save=True,
    workers=4,
    cos_lr=True,
    overlap_mask=False,
    # device=[0,1]
)