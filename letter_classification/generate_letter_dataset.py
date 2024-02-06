import numpy as np
import os
import cv2 as cv
import sys
from dataclasses import dataclass

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
# set this to true if the letter dataset's labels are letters first (0-25) then numbers (26-35)
# new letter datasets should have labels that are numbers first (0-9) then letters (10-35)
OLD_LABELS = False

@dataclass
class LetterBoxInfo:
    x: int
    y: int
    width: int
    height: int
    letter_label: int
    found: int

def get_letter_boxes(label_path, img_shape) -> list[LetterBoxInfo]:
    letter_boxes: list[LetterBoxInfo] = []
    with open(label_path) as f:
        for line in f.readlines():
            label = line.split(' ')
            letter_label = int(label[1])
            box = np.array([float(v) for v in label[4:]])
            box[[0,2]]*=img_shape[1]
            box[[1,3]]*=img_shape[0]
            box[[0,1]] -= box[[2,3]]/2 # adjust xy to be top-left
            x,y,w,h = box.astype(int)
            letter_boxes.append(LetterBoxInfo(x,y,w,h,letter_label, False))
    return letter_boxes

def generate_letter_dataset_from_folder(from_folder, to_folder):
    if from_folder == "":
        from_folder = CURRENT_FILE_PATH + "/fullsize_dataset"
    if to_folder == "":
        to_folder = CURRENT_FILE_PATH + "/letter_dataset"
    for img_file_name in os.listdir(f"{from_folder}/images"):
        img = cv.imread(f"{from_folder}/images/{img_file_name}")
        letter_boxes = get_letter_boxes(f"{from_folder}/labels/{img_file_name.split('.')[0]}.txt", img.shape)
        i = 0
        for box_info in letter_boxes:
            x, y = box_info.x, box_info.y
            w, h = box_info.width, box_info.height
            label = box_info.letter_label
            if OLD_LABELS:
                # convert old labels to new labels
                if label <= 24:
                    label += 10
                else:
                    label -= 25
            with open(f"{to_folder}/labels/{img_file_name.split('.')[0]}_{i}.txt", 'w') as f:
                f.write(str(label))
            cropped_image = img[y:(y+h), x:(x+w)]
            result_image  = cv.resize(cropped_image, (128,128))
            cv.imwrite(f"{to_folder}/images/{img_file_name.split('.')[0]}_{i}.png", result_image)
            i += 1

if __name__ == "__main__":
    from_folder = ""
    to_folder = ""
    if len(sys.argv) > 1:
        from_folder = str(sys.argv[1])
    if len(sys.argv) > 2:
        to_folder = str(sys.argv[2])
    generate_letter_dataset_from_folder(from_folder, to_folder)