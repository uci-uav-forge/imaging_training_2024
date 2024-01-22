from pathlib import Path

#-----CONVERSION PARAMS----------

# The base output directory for the YOLO dataset
TARGET_DIR = Path('/Volumes/SANDISK/YOLO_DATASET/') # Change this to your desired output directory

# The format of the dataset. Either 'bbox' or 'semantic'
DATASET_FORMAT = 'bbox'

# If True, will create a new version of the dataset. Otherwise, will overwrite the existing one
CREATE_NEW_VERSION = False 

#---------------------------

#-------ISSAC PARAMS--------

# The directory containing the ISSAC dataset
DATA_DIR = Path('/Volumes/SANDISK/Issac_data')

# The directory containing the ISSAC train images (.png)
IMAGES_DIR = DATA_DIR / 'train'

# The directory containing the ISSAC semantic masks (.png)
SEMANTIC_MASKS_DIR = DATA_DIR / 'semantic_masks'

# The directory containing the ISSAC semantic labels (.json)
SEMANTIC_LEGEND_DIR = DATA_DIR / 'semantic_labels'

# The directory containing the ISSAC bounding box positions (.npy)
BOXES_DIR = DATA_DIR / 'bounding_box_labels/bbox_labels'

# The directory containing the ISSAC bounding box labels (.json)
BOXES_LEGEND_DIR = DATA_DIR / 'bounding_box_labels/bbox_class_labels'

# --------------------------