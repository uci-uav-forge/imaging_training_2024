import numpy as np
import cv2
from typing import List
from data_types import YoloBbox, YoloLabel, YoloImageData
from yolo_io_types import Task
from data_transformers import BBoxToCropTransformer
import os
import json

# DESIGNED TO BE USED WITH "isaac_data_split_500_2-8-24.zip"

def main():
    task = Task('test') 

    # Create a folder to save the images to. If the folder exists, make a new one (e.g. add a number to the folder name)
    base_folder_name = "data/transformer_output"
    run_num = 1
    while os.path.exists(f"{base_folder_name}_{run_num}"):
        run_num += 1
    master_folder_name = f"{base_folder_name}_{run_num}"
    os.makedirs(master_folder_name)

    # Start processing the data and saving the images
    transformer = BBoxToCropTransformer(min_size=(50, 50), min_padding=10, min_char_overlap=0)
    read_dir = "data/isaac_data_split_500_2-8-24"
    for file in os.listdir(read_dir):
        if file.endswith(".png") and file.startswith("rgb_"):
            id = file.split(".")[0].split("_")[-1]
            print(f"Processing {id}")
            image = cv2.imread(f"{read_dir}/{file}")
            height, width, _ = image.shape
            bboxes_raw = np.load(f"{read_dir}/bounding_box_2d_tight_{id}.npy")
            with open(f"{read_dir}/bounding_box_2d_tight_labels_{id}.json", "r") as f:
                labels_raw = json.load(f) # sample: {"0": {"class": "background"}, "5": {"class": "quartercircle"}, "8": {"class": "rectangle"}, "9": {"class": "semicircle"}, "10": {"class": "n"}, "15": {"class": "z"}}
            # filter bboxes that are not of relevant types based on labels
            yolo_labels: List[YoloLabel] = []
            for bbox in bboxes_raw:
                className = labels_raw[str(bbox[0])]["class"]
                if className in ("circle", "semicircle", "quarter circle", "quartercircle", "quarter_circle", "triangle", "rectangle", "pentagon", "star", "cross") or len(className) == 1:
                    # transform bbox pixel coords to float relative coords
                    # xmin (1), ymin (2), xmax (3), ymax (4) -> x, y, w, h
                    xmin, ymin, xmax, ymax = bbox[1], bbox[2], bbox[3], bbox[4]
                    x = (xmin + xmax) / 2.0 / width
                    y = (ymin + ymax) / 2.0 / height
                    w = (xmax - xmin) / width
                    h = (ymax - ymin) / height
                    yolo_labels.append(YoloLabel(YoloBbox(x, y, w, h), className))
            img_data = YoloImageData(img_id=file, task=task, image=image, labels=yolo_labels)
            transformed_data = list(transformer(img_data))

            # Make a new folder (in the master folder) for this image 
            img_folder_name = f"{master_folder_name}/{id}"
            os.makedirs(img_folder_name)

            # Save og image
            cv2.imwrite(f"{img_folder_name}/original.png", img_data.image)

            # save other images
            for i, data in enumerate(transformed_data):
                cv2.imwrite(f"{img_folder_name}/cropped_{i}.png", data.image)
                # print("Wrote to", f"{img_folder_name}/cropped_{i}.png")
                # print(f'Image ID: {data.img_id}, Classes: {[label.classname for label in data.labels]}')
            # write the labels to a file
            with open(f"{img_folder_name}/labels.txt", "w") as f:
                for i, data in enumerate(transformed_data):
                    f.write(f"Image ID: {data.img_id}, Classes: {[label.classname for label in data.labels]}\n")
    

if __name__ == "__main__":
    main()
