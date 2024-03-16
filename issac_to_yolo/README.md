# Issac To Yolo dataset converter

The name is self-explanatory :)

## Usage
### Initial Config
Edit the params in the ```yolo_config.py``` file as needed. The most important ones are:

 - ```TARGET_DIR```: This is your folder output directory. Within this folder, the script will generate a sub-folder for each dataset gen.
 - ```CREATE_NEW_VERSION```: This flag sets whether or not to overwrite the latest dataset version subfolder (```False```) or make a new subfolder (```True```).
   - If you are overwriting, it will prompt you to confirm.
 - ```TILE_SIZE```: This needs to be set correctly to the size of the tiles you want to produce.
   - Square images of ```TILE_SIZE``` by ```TILE_SIZE``` will be produced (in pixels).
 - ```DATA_DIR```: This is the main folder which has all the data from Issac. It should be the lowest-level folder.
   - (ie. MainFolder --> data1.png, data1.np, data1.json, etc)

### Running the code
 - Ensure the params in ```yolo_config.py``` are correct.
 - Then: ```python3 create_yolo_dataset.py```
 - A progress bar will indicate how much time is left.
 - The dataset will be output in "```TARGET_DIR```/DatasetV#" where # is the dataset version.

