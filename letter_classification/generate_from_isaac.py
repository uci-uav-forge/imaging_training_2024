import numpy as np
import cv2 as cv
import os

""" 
example:
bounding_box_2d_tight_labels_0000.json: 
{"0": {"class": "background"}, "1": {"class": "color_shape"}, 
    "2": {"class": "y"}, "3": {"class": "e"}, "4": {"class": "l"}}
bounding_box_2d_tight_0000.npy:
[(3, 54, 664, 58, 671, 0. )]  one element in array
[0] = 3:   class ("e" from json)
[1] = 54:  x-coord from
[2] = 664: y-coord from
[3] = 58:  x-coord to
[4] = 671: y-coord to
number of letters: number of elements in .npy array with a class belonging to one of the letters
"""
CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

from_path = CURRENT_FILE_PATH + "/isaac_sim_dataset"
to_path = CURRENT_FILE_PATH + "/datasets/dataset"
category = "validation"
start_index = 450
end_index = 500

num_classes = 28  # change this when a more complete dataset is available

# create empty folders (if not yet created) for the letter images (folder name = class label)
for label in range(start_index, end_index):
    os.makedirs(f"{to_path}/{category}/{label}", exist_ok = True) 

for i in range(450, 500):  # range of file numbers of npy file and image
    file_index = f"{i:04d}" # 4 digits for each index (e.g. 0001, 0100)
    boxes = np.load(f"{from_path}/bounding_box_npy/bounding_box_2d_tight_{file_index}.npy")
    img = cv.imread(f"{from_path}/images/rgb_{file_index}.png")
    j = 0
    for box in boxes:
        # box[0] contains the class label
        label = box[0]
        if label >= 2:
            # label 0: background
            # label 1: shape
            # the box contains a letter, so we crop it out
            # remap label so that the first letter label starts from 1
            label -= 2
            # [y_to:y_from, x_to:x_from]
            if box[1] > 0: box[1] -= 1
            if box[2] > 0: box[2] -= 1
            if box[3] < img.shape[0]-1: box[3] += 1
            if box[4] < img.shape[1]-1: box[4] += 1
            cropped_image = img[box[2]:box[4], box[1]:box[3]]
            result_image  = cv.resize(cropped_image, (128,128))
            cv.imwrite(f"{to_path}/{category}/{label}/rbg_{file_index}_{j}.png", result_image)
            j += 1