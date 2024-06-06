from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
import random
from typing import Iterable
from yolo_to_yolo.data_transformers import YoloDataTransformer
from yolo_to_yolo.data_types import YoloImageData
from yolo_to_yolo.yolo_data_pipeline import YoloDataPipeline, DummyTransformer

import cv2
import numpy as np
import math, time


class Augmentation(YoloDataTransformer):
    def __init__(self, probability: float = 0.5):
        self.probability = probability


class RandomFlip(Augmentation):
    def __call__(self, input_data: YoloImageData) -> Iterable[YoloImageData]:
        if random.random() > self.probability:
            yield input_data
        else:
            img = input_data.image
            label = input_data.labels

            #all three types of flip combinations were implemented simultaneously
            img_1, img_2, img_3 = img.copy(), img.copy(), img.copy()

            v_flip = cv2.flip(img_1, 0)
            h_flip = cv2.flip(img_2, 1)
            vh_flip = cv2.flip(img_3, -1)

            v_label = [bbox._replace(location = bbox.location._replace(y = 1-bbox.location.y)) for bbox in label]
            h_label = [bbox._replace(location = bbox.location._replace(x = 1-bbox.location.x)) for bbox in label]
            vh_label = [bbox._replace(location = bbox.location._replace(x = 1-bbox.location.x,
                                                                        y = 1-bbox.location.y)) for bbox in label]
            
            unique_id_1 = input_data.img_id.split("_")[0] + "_" + str(int(time.time()*1e6))[-6:]
            unique_id_2 = input_data.img_id.split("_")[0] + "_" + str(int(time.time()*1e6))[-6:]
            unique_id_3 = input_data.img_id.split("_")[0] + "_" + str(int(time.time()*1e6))[-6:]
            
            flip_1 = YoloImageData(img_id= unique_id_1, task= input_data.task, image=v_flip, labels= v_label)
            flip_2 = YoloImageData(img_id= unique_id_2, task= input_data.task, image=h_flip, labels= h_label)
            flip_3 = YoloImageData(img_id= unique_id_3, task= input_data.task, image=vh_flip, labels= vh_label)

            yield flip_1
            yield flip_2
            yield flip_3

class DirectionalBlur(Augmentation):
    def __call__(self, input_data: YoloImageData) -> Iterable[YoloImageData]:
        if random.random() > self.probability:
            yield input_data
        else:
            for _ in range(0, random.randint(2,5)):
                img = input_data.image.copy()

                kernel_size, rand_angle = random.randrange(3, 31, 2), random.randint(-180, 180)

                kernel = np.zeros((kernel_size, kernel_size))
                kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
                kernel /= kernel_size

                # Apply the rotation to the kernel and apply linear blur
                rotated_kernel = cv2.warpAffine (kernel, cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), rand_angle, 1.0),
                                                (kernel_size, kernel_size))
                apply_photo = cv2.filter2D(img, -1, rotated_kernel)

                unique_id = input_data.img_id.split("_")[0] + "_" + input_data.img_id.split("_")[1]  + str("_") + str(int(time.time()*1e6))[-6:]

                new_data = YoloImageData(
                    img_id=unique_id,
                    task=input_data.task,
                    image=apply_photo,
                    labels=input_data.labels
                )
                yield new_data
        
        
class Contrast(Augmentation):
    def __call__(self, input_data: YoloImageData) -> Iterable[YoloImageData]:
        if random.random() > self.probability:
            yield input_data
        
            for _ in range(0, random.randint(1,2)):

                img = input_data.image.copy()

                #alpha value is calculated based on a gaussian random variable with mu and sigma arbitrary picked.
                mu, sigma = random.choice([1.8, 0.8]), 0.3
                alpha = min( 2, max(0.8, random.gauss(mu, sigma)))
                alpha = 0.8 if alpha == 1 else alpha
                img = cv2.convertScaleAbs(img, alpha= alpha, beta=0)

                unique_id = input_data.img_id.split("_")[0]  + str("_") + input_data.img_id.split("_")[1]  + str("_") + str(int(time.time()*1e6))[-6:]
                new_data = YoloImageData(img_id= unique_id, task= input_data.task, image=img, labels= input_data.labels)

                yield new_data

class Brightness(Augmentation):
    def __call__(self, input_data: YoloImageData) -> Iterable[YoloImageData]:
        if random.random() > self.probability:
            yield input_data
        else:
            for _ in range(0, random.randint(1,2)):
                img = input_data.image.copy()

                #beta is determined by a Ramdom variable with Gaussian distribution. Mu and Sigma can be adjusted accordingly
                mu, sigma = 30, 10
                beta = int(min(255, max(0, random.gauss(mu, sigma))))
                img = cv2.convertScaleAbs(img, alpha= 1 , beta=beta)

                unique_id = input_data.img_id.split("_")[0] + "_" + input_data.img_id.split("_")[1]  + str("_") + str(int(time.time()*1e6))[-6:]
                new_data = YoloImageData(img_id= unique_id, task= input_data.task, image=img, labels= input_data.labels)

                yield new_data
    
class Gaussian(Augmentation):
    def __call__(self, input_data: YoloImageData) -> Iterable[YoloImageData]:
        if random.random() > self.probability:
            yield input_data
        else:
            for _ in range(0, random.randint(2,5)):
                img = input_data.image.copy()

                kernel_size = random.randrange(3, 31, 2)

                output_img = cv2.GaussianBlur( img, (kernel_size, kernel_size), cv2.BORDER_DEFAULT )

                # After applying blur, create individual id based on microseconds
                unique_id = input_data.img_id.split("_")[0] + "_" + input_data.img_id.split("_")[1]  + str("_") + str(int(time.time()*1e6))[-6:]

                new_data = YoloImageData(
                    img_id=unique_id,
                    task=input_data.task,
                    image= output_img,
                    labels=input_data.labels
                )
                yield new_data

        pass

class Rotation(Augmentation):
    def __call__(self, input_data: YoloImageData) -> Iterable[YoloImageData]:
        if random.random() > self.probability:
            yield input_data
        else:
            for _ in range(0, random.randint(2,5)):
                img = input_data.image.copy()
                label = input_data.labels

                rand_angle = random.randint(-180, 180)

                width, height = img.shape[:2]

                rotate_kernel = cv2.getRotationMatrix2D(center = (width//2, height//2), angle = rand_angle, scale = 1)
                rotate_img = cv2.warpAffine(src= img, M = rotate_kernel, dsize = (width, height), borderMode = cv2.BORDER_WRAP)

                vh_labels = []
                for bbox in label:
                    coord = (bbox.location.x, bbox.location.y)
                    new_coord = self._rotate_pt(coord, rand_angle)

                    #the rotation math has been verified by drawing circles around the new coordinate points after reversing the bbox point normalization
                    #cv2.circle(rotate_img, (int(new_coord[0]*width), int(new_coord[1]*height)), 30, (255,0,0), 5)

                    new_label = bbox._replace(location = bbox.location._replace(x = new_coord[0], y = new_coord[1]))
                    vh_labels.append(new_label)

                unique_id = input_data.img_id.split("_")[0] + "_" + input_data.img_id.split("_")[1]  + str("_") + str(int(time.time()*1e6))[-6:]

                rot_img = YoloImageData(
                    img_id= unique_id, 
                    task= input_data.task, 
                    image= rotate_img, 
                    labels= vh_labels)
                
                yield rot_img

    def _rotate_pt(self, coordinates:tuple, angle: int):
        rad = np.deg2rad(angle)
        new_x = (coordinates[0] - 0.5 ) * math.cos( rad ) + (coordinates[1] - 0.5) * math.sin( rad ) + 0.5
        new_y = -(coordinates[0] - 0.5 ) * math.sin( rad ) + (coordinates[1] - 0.5) * math.cos( rad ) + 0.5
        return (new_x, new_y)

class Exposure(Augmentation):
    def __call__(self, input_data: YoloImageData) -> Iterable[YoloImageData]:
        #gamma correction: adjustment to gamma will be based on a random variable gaussian distribution
        if random.random() > self.probability:
            yield input_data
        else:
            for _ in range(4,9):
                img = input_data.image.copy()

                #exposure is a gamma related operation with the formula being the image's normalized pixel value raised to the power of 1/gamma. normalization is undone to create the new image.
                mu = random.choice([0.8, 1.5])
                sigma = 0.2
                gamma = min( 2, max(0.5, random.gauss(mu, sigma)))
                gamma_img = (np.power(img/255, 1/gamma) * 255 ).astype("uint8")

                unique_id = input_data.img_id.split("_")[0] + "_" + input_data.img_id.split("_")[1]  + str("_") + str(int(time.time()*1e6))[-6:]

                expose_img = YoloImageData(
                    img_id= unique_id, 
                    task= input_data.task, 
                    image= gamma_img, 
                    labels= input_data.labels)

                yield expose_img


class Saturation(Augmentation):
    def __call__(self, input_data: YoloImageData) -> Iterable[YoloImageData]:
        if random.random() > self.probability:
            yield input_data
        else:
            for _ in range(4,9):
                img = input_data.image.copy()

                #saturation is done by manually adjusting the saturation channel in the image's hsv color channel
                hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype("float32")
                (h,s,v) = cv2.split(hsv_img)

                #use a gaussian RV to pick a saturation factor, mu and sigma are arbitrary set
                mu, sigma = 1.5, 0.4
                s_factor = min( 2, max(0.6, random.gauss(mu, sigma)))
                s = np.clip(s * s_factor, 0,255)

                hsv_img = cv2.merge([h,s,v])
                img = cv2.cvtColor(hsv_img.astype("uint8") , cv2.COLOR_HSV2BGR)

                unique_id = input_data.img_id.split("_")[0] + "_" + input_data.img_id.split("_")[1]  + str("_") + str(int(time.time()*1e6))[-6:]
                expose_img = YoloImageData(
                    img_id= unique_id, 
                    task= input_data.task, 
                    image= img, 
                    labels= input_data.labels)

                yield expose_img


                

if __name__ == '__main__':
    """
    Example usage
    """
    input_dir = "/mnt/c/Users/kirva/Desktop/Project_Design/Project_UAV/uavf_2024/imaging_training_2024/YOLO_DATASET/DATASETv1"
    output_dir = "/mnt/c/Users/kirva/Desktop/Project_Design/Project_UAV/uavf_2024/imaging_training_2024/YOLO_DATASET/DATASET_mot"
    
    #probabilities are arbitrary set and pipeline allows stacking of multiple augmentations
    pipeline = YoloDataPipeline(pipeline= [Rotation(0.2), DirectionalBlur(0.3)])
    
    pipeline.apply_to_dir(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir)
    )