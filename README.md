# imaging_training_2024

## Synthetic data

### Generation
Currently, the code only works on Vince's computer. The many attempts to run it elsewhere have thus failed. 
It's available at https://github.com/uci-uav-forge/UAV-Forge-Imaging-Sim for reference.

### Retrieval
The generated data can be downloaded from https://drive.google.com/drive/folders/1bFDhUh6WVlGPChKcFt_PdUnMzQb1qoSw.
The relavent files are `train.tar.gz`, `semantic_segmentation_labels_.tar.gz`, `semantic_masks.tar.gz`, `bbox_location_labels.tar.gz`, and `bbox_location_labels.tar.gz`. Note that the last two are nested in the `bounding_box_labels` folder. 

### Usage

#### Interface Setup
The interface written in `sim_data_interface` assumes that the folders are flattened, such that the structure is:

```
data [self-defined]
├── bbox_class_labels
├── bbox_labels
├── semantic
├── semantic_segmentation_labels
├── train
```

The default names can be changed in `sim_data_interface/config.py` and can also be overridden in the `Dataset` constructor. Note that the above names are not the same as the archive names; they're the names on the directories inside the archive.

#### Interface Usage
The interface is written in `sim_data_interface/dataset.py`. The `Dataset` class is the main interface. There is an example at the bottom of the file.

Latest dataset link (january): https://drive.google.com/drive/folders/1bFDhUh6WVlGPChKcFt_PdUnMzQb1qoSw

## Troubleshooting

### `_imagingft` Import Error
  - Install libfreetype: `sudo apt-get install libfreetype6-dev`
  - Create a a new environment
  - Reinstall the packages with `pip install -r requirements.txt --no-binary :all`
    - `--no-binary :all` forces recompilation, which can help.

### YOLO: Dataset Not Found
- Check the path of the subfolders in data.yaml.
  - If `path` is defined, it is relative to the CWD, not the file itself. Use absolute paths to be sure.
  - The other path parameters are relative to the `path` parameter, if defined.
