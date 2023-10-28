from ultralytics import YOLO

task='segmentation'
if task=='segmentation':
    model = YOLO('yolov8n-seg.yaml')
elif task=='detection':
    model = YOLO('yolov8n.yaml')
else:
    raise NotImplementedError
    
model.train(
    data='shapes_2023.yaml', 
    epochs=5, 
    save=True,
    workers=4,
    cos_lr=True,
    overlap_mask=False,
    # device=[0,1]
)