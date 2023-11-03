# Generating letter dataset from isaac sim fullsize dataset

`generate_from_isaac.py` generates a folder of letter dataset from a specified folder and range of isaac sim dataset. `tiny_isaac_example_dataset` contains an small example of the letter dataset generated from the fullsize dataset.

## Required folder structure for isaac sim fullsize dataset:
To run the script without errors, the fullsize dataset folder should follow this structure:<br/>
    ├── bounding_box_npy <br/>
    │ &emsp; └──bounding_box_2d_tight_xxxx.npy <br/>
    └── images <br/>
    &emsp;&emsp; └── rbg_xxxx.png <br/>

## Running the script
Before running `generate_from_isaac.py`, there are 5 values at the top you need to change:
* `from_path`: the path of the folder containing the issac sim dataset (following the structure above).
* `to_path`: the path of the folder to create the letter dataset in.
* `category`: enter `"train"`. `"test"`, or `"validation"` depending on what you'll use the dataset for when training the letter model.
* `start_index` and `end_index`: since the fullsize dataset is numbered (e.g. `rbg_0001.png`), these indices indicate the range from the fullsize dataset you're using to create the letter dataset from. </br>

## Folder structure of the letter dataset after running
After running the script, here's the folder structure of the letter dataset: </br>
    ├── test <br/>
    │ &emsp; ├──0 <br/>
    │ &emsp; │ &emsp; └──rgb_xxxx_x.png <br/>
    │ &emsp; ├──1 <br/>
    │ &emsp; │ &emsp; └──rgb_xxxx_x.png <br/>
    │ &emsp; └──2 <br/>
    │ &emsp;&emsp;&emsp; └──rgb_xxxx_x.png <br/>
    ├── train <br/>
    │ &emsp; ├──0 <br/>
    │ &emsp; │ &emsp; └──rgb_xxxx_x.png <br/>
    │ &emsp; ├──1 <br/>
    │ &emsp; │ &emsp; └──rgb_xxxx_x.png <br/>
    │ &emsp; └──2 <br/>
    │ &emsp;&emsp;&emsp; └──rgb_xxxx_x.png <br/>
    └── validation <br/>
    &emsp;&emsp; ├──0 <br/>
    &emsp;&emsp; │ &emsp; └──rgb_xxxx_x.png <br/>
    &emsp;&emsp; ├──1 <br/>
    &emsp;&emsp; │ &emsp; └──rgb_xxxx_x.png <br/>
    &emsp;&emsp; └──2 <br/>
    &emsp;&emsp;&emsp;&emsp; └──rgb_xxxx_x.png <br/>

## To-Do
Generate a complete dataset with all letter classes, remap the class labels, and train the letter model with the complete dataset. 

# (For Old Dataset)
## Generating letter dataset from fullsize dataset
`generate_letter_dataset.py` generates a set of letter dataset from a set of fullsize dataset by cropping out the bounding boxes specified in the fullsize dataset's labels.

When running the script, the folder of the existing dataset and the folder to create the letter dataset in can be specified.  Otherwise, the default folders are `fullsize_dataset` and `letter_dataset` within this current folder.

Here's an example to specify the folders when running the script in the terminal (the current working directory is this folder, `letter_classification`): <br/>
`python3 generate_letter_dataset.py fullsize_dataset letter_dataset`

After running the script, the images should be created inside the `images` folder in the letter dataset folder, and the labels should be created inside the `labels` folder.

## Folder structure for the dataset required before running the script
To run the script without errors, both folders specified should follow this structure and have the images and labels folders created (can be empty inside):<br/>
    ├── images <br/>
    └── labels <br/>

## To do:
Since the difference in between the background and the shape's background can affect the letter model's training, the script crops out the center 50% of the bounding box (e.g. (x+w/4):(x+(3*w/4)) instead of the entire bounding box to eliminate the background outside the shape.  However, some parts outside of the shape are still in the image, so this is an area of the script that can be improved.


