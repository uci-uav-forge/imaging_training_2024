from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from yolo_io import YoloReader, YoloWriter
from yolo_io_types import PredictionTask, Task

def prompt_for_file(prompt_text: str) -> str:
    """
    Opens a file dialog using Tkinter, allowing the user to select a YAML file.

    Args:
        prompt_text (str): The text to display in the dialog window
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title=prompt_text, filetypes=[("YAML files", "*.yaml")])
    root.destroy()
    return file_path

def prompt_for_directory(prompt_text: str) -> str:
    """
    Opens a folder dialog using Tkinter, allowing the user to select a directory.

    Args:
        prompt_text (str): The text to display in the dialog window
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    directory = filedialog.askdirectory(title=prompt_text)
    root.destroy()
    return directory

def filter_and_copy_files(reader: YoloReader, writer: YoloWriter, total_files_to_copy: int) -> None:
    """
    Filters and copies files across train, test, and val directories based on their proportionate counts.

    Args:
        reader (YoloReader): The reader instance to read the YOLO dataset
        writer (YoloWriter): The writer instance to write the filtered YOLO dataset
        total_files_to_copy (int): The total number of files to copy across the subsets
    """
    # Determine the number of images in each subset
    tasks_to_process = (Task.TRAIN, Task.VAL, Task.TEST)
    subset_sizes = {task: len(list(reader.parent_dir.joinpath(reader.descriptor.get_image_and_labels_dirs(task).images).glob("*.png"))) for task in tasks_to_process}
    total_images = sum(subset_sizes.values())

    # Calculate how many images to copy from each subset
    files_to_copy = {task: round((subset_sizes[task] / total_images) * total_files_to_copy) for task in tasks_to_process}

    for task in tasks_to_process:
        image_data_gen = reader.read(tasks=(task,))
        for _, yolo_image_data in zip(range(files_to_copy[task]), image_data_gen):
            writer.write([yolo_image_data])

def main():
    input_yaml = Path(prompt_for_file("Select the YOLO dataset YAML file"))
    if not input_yaml:
        print("No YAML file selected. Exiting.")
        return

    output_dir = Path(prompt_for_directory("Select Output Directory"))
    if not output_dir:
        print("No output directory selected. Exiting.")
        return

    try:
        total_files_to_copy = int(input("Enter the total number of files to copy: "))
    except ValueError:
        print("Invalid number entered. Exiting.")
        return

    # Create reader and writer instances
    reader = YoloReader(yaml_path=input_yaml, prediction_task=PredictionTask.DETECTION)
    writer = YoloWriter(out_dir=output_dir, prediction_task=PredictionTask.DETECTION, classes=reader.classes)

    print("Copying files...")
    filter_and_copy_files(reader, writer, total_files_to_copy)
    print("Done!")

if __name__ == "__main__":
    main()
