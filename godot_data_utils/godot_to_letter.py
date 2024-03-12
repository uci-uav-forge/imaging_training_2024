# takes dataset with masks generated with godot and turns it into a dataset with labels for YOLOv8
from data_gen_utils import get_polygon, get_letter_box, LetterBoxInfo
import os
import cv2
import json
import numpy as np
from tqdm import tqdm


def preprocess_img(img):
    # blur image with random kernel size
    kernel_size = 3 + 2*np.random.randint(0, 2)
    if np.random.randint(0,2)==0:
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    else:
        img = cv2.boxFilter(img, -1, (kernel_size, kernel_size))
    # add random noise with random variance
    variance = np.random.randint(0, 10)
    img = img + np.random.normal(0, variance, img.shape)
    # clamp values to 0-255
    np.clip(img, 0, 255, out=img)
    return img

def gen_img(num, num_images, input_dir, output_dir, letter_to_categories):
    index = 0
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
    for mask_file_name in os.listdir(f"{input_dir}/masks/{num}"):
        mask_path = f"{input_dir}/masks/{num}/{mask_file_name}"
        # mask_file_name example: semicircle,X,160-83-170,154-22-90_1.png
        info = mask_file_name.split("_")[0].split(",")
        # ignore images without shapes (lke person_0.png)
        if len(info) == 1:
            continue
        letter = info[1]
        mask = cv2.imread(mask_path)
        polygon = get_polygon(mask)

        if len(polygon) <= 2:
            if os.getenv("VERBOSE") is not None:
                tqdm.write(f"no polygon found for {mask_path}")
            return 
        normalized_polygon = polygon / np.array([mask.shape[1], mask.shape[0]])
        letter_box: LetterBoxInfo = get_letter_box(normalized_polygon, img.shape, letter_to_categories[letter])
        # crop image
        x, y = letter_box.x, letter_box.y
        w, h = letter_box.width, letter_box.height
        # ignore partial shapes/letters
        if w < 80 or h < 80:
            continue
        cropped_img = img[y:y+h, x:x+w]
        result_img  = cv2.resize(cropped_img, (128,128))
        # write image to the letter label's folder
        cv2.imwrite(f"{output_dir}/{split_name}/{letter_box.letter_label}/image{num}_{index}.png", result_img)
        index += 1


def main():
    user = os.environ["USER"]
    datagen_dir = os.path.dirname(os.path.abspath(__file__))
    categories_to_letter = json.load(open(f"{datagen_dir}/letter_labels.json","r"))
    letter_to_categories = {letter:category for category, letter in categories_to_letter.items()}
    # for linux
    # input_dir = f"/home/{user}/.local/share/godot/app_userdata/forge-godot/godot_data"
    # for windows (change user name)
    input_dir = "/mnt/c/Users/sch90/AppData/Roaming/Godot/app_userdata/forge-godot/godot_data"
    output_dir = f"{datagen_dir}/letter_data"
    os.makedirs(output_dir, exist_ok=True)
    for split_name in ["train", "validation", "test"]:
        os.makedirs(f"{output_dir}/{split_name}", exist_ok=True)
        for i in range(0, 36):
            os.makedirs(f"{output_dir}/{split_name}/{i}", exist_ok=True)
    num_images = len(os.listdir(f"{input_dir}/images"))

    for i in tqdm(range(num_images)):
        gen_img(i, num_images, input_dir, output_dir, letter_to_categories)
if __name__ == "__main__":
    main()