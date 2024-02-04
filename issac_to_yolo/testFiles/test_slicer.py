from PIL import Image, ImageFont, ImageDraw
import numpy as np 
import math

def determineOverlap(section : int, total_length : int) -> tuple():
    amount = math.ceil(total_length / section)
    overlap = int(((amount * section) - total_length) / (amount-1))
    return (amount, overlap)

def sliceImage(image_arr : np.array, tileSize : int) -> list[Image.Image]:
    height, width, channels = image_arr.shape
    height_amount, height_overlap = determineOverlap(tileSize, height)
    width_amount, width_overlap = determineOverlap(tileSize, width)

    tiles = []
    for i in range(height_amount):
        for j in range(width_amount):

            y_start = i*(tileSize-height_overlap)
            y_end = y_start+tileSize

            x_start = j*(tileSize-width_overlap)
            x_end = x_start+tileSize

            tiles.append(Image.fromarray(image_arr[y_start:y_end, x_start:x_end]))

    return tiles


if __name__ == "__main__":
    # Load image
    image = Image.open("testImages/2205.png")
    image_arr = np.array(image)

    # Slice image
    tileSize = 512

    images = sliceImage(image_arr, tileSize)

    # # Display image 
    for image in images:
        image.show()