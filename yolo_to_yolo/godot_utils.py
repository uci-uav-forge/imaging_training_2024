import cv2
import numpy as np
import os
from dataclasses import dataclass
from .data_types import YoloBbox

def get_polygon(shape_img: cv2.Mat) -> np.ndarray:
    '''
    Returns the enclosing polygon of the shape in the image. The polygon is a list of points, each point being a list of 2 coordinates.
    '''
    im = cv2.cvtColor(shape_img, cv2.COLOR_BGR2GRAY)
    im = cv2.threshold(im, 253, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        return np.array([]) 
    if len(contours)>1:
        if os.getenv("VERBOSE") is not None:
            print("Warning: multiple contours found")
        # combine contours and return convex hull
        contours = np.concatenate([c.reshape(-1,2) for c in contours])
        contours = cv2.convexHull(contours)
        return contours.reshape(-1,2)
    return np.array(contours[0]).reshape(-1,2)

@dataclass
class LetterBoxInfo:
    x: int
    y: int
    width: int
    height: int
    letter_label: int

def get_letter_box(polygon_points: np.ndarray, img_shape: [], letter_label: str) -> LetterBoxInfo:
    # polygon_points = [[x, y], [x, y], [x,y], ...]
    # returns the bounding box for the shape containing the letter
    x_min, x_max, y_min, y_max = None, None, None, None
    for point in polygon_points:
        x = point[0]
        y = point[1]
        if x_min == None or x < x_min:
            x_min = x
        if x_max == None or x > x_max:
            x_max = x
        if y_min == None or y < y_min:
            y_min = y
        if y_max == None or y > y_max:
            y_max = y
    x = x_min * img_shape[0]
    y = y_min * img_shape[1]
    width = (x_max - x_min) * img_shape[0]
    height = (y_max - y_min) * img_shape[1]
    letter_box = LetterBoxInfo(int(x), int(y), int(width), int(height), letter_label)
    return letter_box
                
                
def give_normalized_bounding_box( norm_polygon_array: np.ndarray) -> YoloBbox:
    '''Returns bounding box as proportion of image dimensions, x_center,y_center,w,h'''

    x_coord = norm_polygon_array[:,0]
    y_coord = norm_polygon_array[:,1]

    if len(x_coord) == 0 or len(y_coord) == 0:
        # Handle the case where one or both arrays are empty
        return None

    min_x, min_y = np.min(x_coord), np.min(y_coord)
    max_x, max_y = np.max(x_coord), np.max(y_coord)
    w = max_x - min_x
    h = max_y - min_y
    x = (max_x+min_x)/2
    y = (max_y+min_y)/2

    return YoloBbox(x, y, w, h)
    

def preprocess_img(img):
    # only augment half of them (why? I made this up on a whim.)
    if np.random.randint(0,2) == 0:
        # blur image with random kernel size
        kernel_size = 3 + 2*np.random.randint(0, 4)
        if np.random.randint(0,2)==0:
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        else:
            img = cv2.boxFilter(img, -1, (kernel_size, kernel_size))
        # add random noise with random variance
        variance = np.random.randint(2, 20)
        img = img + np.random.normal(0, variance, img.shape)
        # clamp values to 0-255
        np.clip(img, 0, 255, out=img)
    return img
