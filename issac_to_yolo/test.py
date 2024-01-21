from sim_data_interface import ImageOverlay, Dataset
import cv2

"""
Example
"""
import matplotlib.pyplot as plt

dataset = Dataset()
id = "0007"

img = dataset.get_image(id)[:, :, ::-1] # BGR to RGB
img_with_mask = dataset.get_image(id, overlay=ImageOverlay.SEMANTIC_MASK)[:, :, ::-1]
img_with_boxes = dataset.get_image(id, overlay=ImageOverlay.BOUNDING_BOXES_WITH_LABELS)[:, :, ::-1]

# Display
figs, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(img)
axs[0].set_title("Image")

axs[1].imshow(img_with_mask)
axs[1].set_title("Image with semantic mask")

axs[2].imshow(img_with_boxes)
axs[2].set_title("Image with bounding boxes")

plt.show()
#figs.savefig("example.png", dpi=600, bbox_inches='tight')