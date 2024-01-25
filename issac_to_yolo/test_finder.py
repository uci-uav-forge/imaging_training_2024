from pathlib import Path


SEARCH_DIR = Path('/Volumes/SANDISK/Issac_data/test_searcher/')

def get_file_id(path_to_file : Path):
    # RGB files are named like this: 'rgb_000000.png'
    # Bbox class label files are named like this: 'bounding_box_2d_tight_000000.json'
    # Bbox legend files are named like this: 'bounding_box_2d_tight_000000.npy'
    # Semantic mask files are named like this: 'semantic_segmentation_000000.png'
    # Semantic legend files are named like this: 'semantic_segmentation_labels_000000.json'
    # Thus the file id is the number after the underscore
    return path_to_file.stem.split('_')[-1]

def get_file_by_id(id, type, root_dir : Path = None):
    # Returns the file with the given id from the given root directory
    # The file id is the number after the underscore
    id = str(id)
    if root_dir is None:
        root_dir = SEARCH_DIR
        if type == 'rgb':
            files = list(root_dir.glob('rgb*.png'))
        elif type == 'semantic':
            files = list(root_dir.glob('semantic_segmentation*.png'))
        elif type == 'semantic_legend':
            files = list(root_dir.glob('semantic_segmentation_labels*.json'))
        elif type == 'bbox_legend':
            files = list(root_dir.glob('bounding_box*.json'))
        elif type == 'bbox_pos':
            files = list(root_dir.glob('bounding_box*.npy'))
    else:
        files = list(root_dir.glob('[!.]*'))

    for file in files:
        if get_file_id(file) == id:
            return file
    return None

if __name__ == '__main__':
    #print(get_file_by_id('0000', 'semantic_legend'))
    #print(get_file_by_id('0000', 'semantic'))
    print(get_file_by_id('0000', 'bbox_pos'))