import numpy as np
import cv2
from data_types import YoloBbox, YoloLabel, YoloImageData
from yolo_io_types import Task
from data_transformers import BBoxToCropTransformer

# NOTE: If you change the bboxes and the cv2 imshow window thingies have large gray chunks, that's just a cv2 thing that happens if the image is too small. You can save it to a file instead to see it properly.

def create_test_image():
    image = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)

    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), 2)  # Single shape (no character inside)
    cv2.rectangle(image, (220, 220), (235, 235), (0, 255, 0), 2) # Random character not inside any shape

    cv2.rectangle(image, (250, 250), (350, 350), (255, 0, 0), 2)  # Shape
    cv2.rectangle(image, (275, 275), (325, 325), (0, 255, 0), 2)  # Character inside shape

    cv2.rectangle(image, (50, 250), (150, 350), (255, 0, 0), 2)   # Shape
    cv2.rectangle(image, (100, 300), (180, 380), (0, 255, 0), 2)  # Character partially overlapping shape

    cv2.rectangle(image, (250, 50), (350, 150), (255, 0, 0), 2)   # Shape
    cv2.rectangle(image, (300, 100), (380, 180), (0, 255, 0), 2)  # Character partially overlapping shape

    return image

def resize_image(image, min_display_size=(100, 100)):
    height, width = image.shape[:2]
    new_height, new_width = max(height, min_display_size[0]), max(width, min_display_size[1])
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return resized_image

def main():
    image = create_test_image()
    task = Task('test') 
    labels = [
        YoloLabel(location=YoloBbox(x=0.3, y=0.3, w=0.2, h=0.2), classname="shape"), # Single shape
        YoloLabel(location=YoloBbox(x=0.455, y=0.455, w=0.03, h=0.03), classname="character"), # Random character

        YoloLabel(location=YoloBbox(x=0.6, y=0.6, w=0.2, h=0.2), classname="shape"),
        YoloLabel(location=YoloBbox(x=0.6, y=0.6, w=0.1, h=0.1), classname="character"),

        YoloLabel(location=YoloBbox(x=0.2, y=0.6, w=0.2, h=0.2), classname="shape"),
        YoloLabel(location=YoloBbox(x=0.28, y=0.68, w=0.16, h=0.16), classname="character"),

        YoloLabel(location=YoloBbox(x=0.6, y=0.2, w=0.2, h=0.2), classname="shape"),
        YoloLabel(location=YoloBbox(x=0.68, y=0.28, w=0.16, h=0.16), classname="character"),
    ]
    img_data = YoloImageData(img_id="test_image", task=task, image=image, labels=labels)

    transformer = BBoxToCropTransformer(min_size=(50, 50), min_padding=10)
    transformed_data = list(transformer(img_data))

    # show og image
    cv2.imshow('Original Image', img_data.image)

    # show cropped images
    for i, data in enumerate(transformed_data):
        resized_cropped_image = resize_image(data.image)
        cv2.imshow(f'Cropped Image {i}', resized_cropped_image)
        print(f'Image ID: {data.img_id}, Classes: {[label.classname for label in data.labels]}')

    print("\n\nThe cropped images should each either have just a blue rectangle (shape) or both a blue and green rectangle (shape and character) inside it.")
    print("It is expected that there is one random green box, this is to text excluding characters not inside shapes.")
    print("\n\n[Press any key (while focusing one of the output windows) to close the windows]")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()