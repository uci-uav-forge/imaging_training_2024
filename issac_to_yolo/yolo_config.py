from pathlib import Path

#-----CONVERSION PARAMS----------

# The base output directory for the YOLO dataset
TARGET_DIR = Path('/home/eesh/forge/YOLO_DATASET/') # Change this to your desired output directory

# The format of the dataset. Either 'bbox' or 'semantic' # currently only bbox is supported
DATASET_FORMAT = 'bbox'

# If True, will create a new version of the dataset. Otherwise, will overwrite the latest existing one
CREATE_NEW_VERSION = False 

TILE_SIZE = 640 # The size of the tiles to be created

DEBUG = False # If True, will reduce conversion to 50 images and produce print statements
#---------------------------

#-------ISSAC PARAMS--------

# The directory containing the ISSAC dataset
DATA_DIR = Path('/home/eesh/forge/issac_datasets/minh-0208/main/sub_main')
# --------------------------