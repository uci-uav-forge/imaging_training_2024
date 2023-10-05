from ultralytics import YOLO

task='segmentation'
if task=='segmentation':
    model = YOLO('yolov8n-seg.yaml')
elif task=='detection':
    model = YOLO('yolov8n.yaml')
    
model.train(
    data='forge10k.yaml', 
    epochs=100, 
    save=True,
    save_period=10,
    workers=4,
    cos_lr=True,
    overlap_mask=False,
    # device=[0,1]
)