# Generating letter dataset from fullsize dataset
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


