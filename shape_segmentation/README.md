# Training with a custom dataset

TODO: write this secton. If you need to work on this rn, just find a time to get Eric to walk you through it. Everything is techincally available on uavf_2023 and godot_data_gen repos from last year but it's hella undocumented.

# Other random notes

It was a pain in the ass to find where ultralytics looks for datasets. Turns out it's under

`(your python path)/site-packages/ultralytics/yolo/data/datasets`. 

As an example of the full path, here's what it looks like on Eric's computer: 

`/home/holden/miniconda3/envs/yolo/lib/python3.8/site-packages/ultralytics/yolo/data`.

As an aside, there is a dataset definition in there called "VisDrone" which sounded like it might be useful, but the images in the dataset aren't straight-down birds-eye view like ours. The camera position is like 20-30 ft in the air and looking sideways/down. There's also another one called "xView" which is satellite images at a macro scale (At least, I think it's macro scale, based on their github: https://github.com/DIUx-xView).

A lot of the documentation for yolo is on the yolov5 repo: https://github.com/ultralytics/yolov5/wiki, not the ultralytics main repo where the ultralytics package is from