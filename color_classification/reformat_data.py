from PIL import Image
import numpy as np
import os

photos_path = './sim_data/train'
labels_path = './sim_data/bbox_labels'

photo_files = [f for f in os.listdir(photos_path) if f.endswith('.png')]

for photo_file in photo_files[0:100]:
    photo_path = os.path.join(photos_path, photo_file)
    photo = Image.open(photo_path)

    bbox_file = 'bounding_box_2d_tight_' + photo_file[4:].replace('.png', '.npy')
    bbox_path = os.path.join(labels_path, bbox_file)
    bounding_boxes = np.load(bbox_path)

    print(bounding_boxes)
    for i, bbox in enumerate(bounding_boxes):
        try:
            label, x_min, y_min, x_max, y_max, occ_rat = bbox
            if label == 0:
                continue
            # x_min = int(x_center - length / 2)
            # y_min = int(y_center - width / 2)
            # x_max = int(x_center + length / 2)
            # y_max = int(y_center + width / 2)

            cropped_photo = photo.crop((x_min, y_min, x_max, y_max))

            new_photo_file = f'cropped_bbox_{i}_{photo_file}'
            new_photo_path = os.path.join('./output', new_photo_file)
            cropped_photo.save(new_photo_path)
        except Exception:
            pass
