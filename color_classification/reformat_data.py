from PIL import Image
import numpy as np
import os
import math

photos_path = '../godot_data_utils/data/images/test'
labels_path = '../godot_data_utils/data/all_box_labels/test'
colors = list({
    "red":(200,40,0),
    "green": (53,194,41),
    "blue": (41,87,194),
    "orange" : (217,101,13),
    "purple": (127,41,194),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'brown': (165, 42, 42),
}.keys())
if not os.path.exists("./output"):
    os.mkdir("./output")
    os.mkdir("./output/data")
    
photo_files = [f for f in os.listdir(photos_path) if f.endswith('.png')]
index = 0
with open("./output/labels.txt", "w") as labels:
    labels.write("file, letter_color, shape_color\n")
    for photo_file in photo_files:
        print(photo_file)
        photo_path = os.path.join(photos_path, photo_file)
        photo = Image.open(photo_path)

        bbox_file = photo_file.replace('.png', '.txt')
        bbox_path = os.path.join(labels_path, bbox_file)

        with open(bbox_path, "r") as f:
            for bbox in f.readlines():
                print("b", bbox)
                if len(bbox) == 0:
                    continue
                if "person" in bbox:
                    continue
                label, _ , shape_color, letter_color, x_center, y_center, x_len, y_len= bbox.split(" ")
                # x_min = int(x_center - length / 2)
                # y_min = int(y_center - width / 2)
                # x_max = int(x_center + length / 2)
                # y_max = int(y_center + width / 2)
                cropped_photo = photo.crop(((float(x_center)-float(x_len)/2) * 640, (float(y_center)-float(y_len)/2) * 640, (float(x_center)+float(x_len)/2)*640, (float(y_center)+float(y_len)/2) *640)).resize((128,128))

                new_photo_file = f'{index}.png'
                new_photo_path = os.path.join('./output/data/', new_photo_file)
                labels.write(f"/data/{new_photo_file}, {colors.index(letter_color)}, {colors.index(shape_color)}\n")
                index+= 1

                # Resize to 128x128
                cropped_photo.save(new_photo_path)