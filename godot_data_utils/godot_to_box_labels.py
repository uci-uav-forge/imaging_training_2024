# takes dataset with masks generated with godot and turns it into a dataset with labels for YOLOv8
from data_gen_utils import get_polygon, preprocess_img, give_normalized_bounding_box
import os
import cv2
import json
import numpy as np
from tqdm import tqdm

user = os.environ["USER"]
# for linux
INPUT_DIR = f"/home/{user}/.local/share/godot/app_userdata/forge-godot/godot_data_1710280018"
# for windows (change username)
# input_dir = "/mnt/c/Users/sch90/AppData/Roaming/Godot/app_userdata/forge-godot/godot_data"

SHAPES = [
 "circle",
 "semicircle",
 "quartercircle",
 "triangle",
 "rectangle",
 "pentagon",
 "star",
 "cross",
 "person"
]

def gen_img(num, num_images, input_dir, output_dir, shapes_to_categories):
    if int(num)<0.85*num_images:
        split_name = "train"
    elif int(num)<0.95*num_images:
        split_name = "validation"
    else:
        split_name = "test" 
    img = cv2.imread(f"{input_dir}/images/image{num}.png")
    if img is None:
        tqdm.write(f"image read error for {input_dir}/images/image{num}.png")
        return
    img = preprocess_img(img)
    file_contents = ""
    shape_file_contents = ""
    for mask_file_name in os.listdir(f"{input_dir}/masks/{num}"):
        mask_path = f"{input_dir}/masks/{num}/{mask_file_name}"
        labels = mask_file_name.split("_")[0].split(",")
        mask = cv2.imread(mask_path)
        polygon = get_polygon(mask)

        if len(polygon) <= 2:
            if os.getenv("VERBOSE") is not None:
                tqdm.write(f"no polygon found for {mask_path}")
            return 
        normalized_polygon = polygon / np.array([mask.shape[1], mask.shape[0]])
        
        bbox_str = give_normalized_bounding_box(normalized_polygon)
        file_contents+=f"{' '.join(labels)} {bbox_str}\n"
        shape_file_contents+=f"{SHAPES.index(labels[0])} {bbox_str}"
    with open(f"{output_dir}/all_box_labels/{split_name}/image{num}.txt", "w") as f:
        f.write(file_contents)
    with open(f"{output_dir}/shape_box_labels/{split_name}/image{num}.txt", "w") as f:
        f.write(shape_file_contents)

def main():
    datagen_dir = os.path.dirname(os.path.abspath(__file__))
    categories_to_shapes = json.load(open(f"{datagen_dir}/shape_name_labels.json","r"))
    shapes_to_categories = {shape:category for category, shape in categories_to_shapes.items()}
    output_dir = f"{datagen_dir}/data_2"
    os.makedirs(output_dir, exist_ok=True)
    for split_name in ["train", "validation", "test"]:
        os.makedirs(f"{output_dir}/all_box_labels/{split_name}", exist_ok=True)
        os.makedirs(f"{output_dir}/shape_box_labels/{split_name}", exist_ok=True)
        os.makedirs(f"{output_dir}/images/{split_name}", exist_ok=True)
    num_images = len(os.listdir(f"{INPUT_DIR}/images"))

    for i in tqdm(range(num_images)):
        gen_img(i, num_images, INPUT_DIR, output_dir, shapes_to_categories)
if __name__ == "__main__":
    main()