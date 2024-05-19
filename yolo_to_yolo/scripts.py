import os
import shutil
import re
from tkinter import Tk, filedialog

def prompt_for_directory(prompt_text: str) -> str:
    """
    Opens a folder dialog using Tkinter, allowing the user to select a directory.
    
    Args:
    prompt_text (str): The title of the dialog window.
    
    Returns:
    str: The path to the selected directory.
    """
    root = Tk()
    root.withdraw()  # Hide the main window
    directory = filedialog.askdirectory(title=prompt_text)
    root.destroy()
    return directory

def copy_files(input_dir: str, output_dir: str, end_number: int, even_only: bool = False) -> None:
    """
    Copies files from the input directory to the output directory based on specified conditions.
    
    Args:
    input_dir (str): The directory to copy files from.
    output_dir (str): The directory to copy files to.
    end_number (int): Maximum file number to consider for copying.
    even_only (bool): Whether to copy only files with even numbers (default is False).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Regex pattern to match the number at the end of filenames after the last underscore
    pattern = re.compile(r'.*_([0-9]+)\.')
    
    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            file_number = int(match.group(1))
            if file_number <= end_number and (not even_only or file_number % 2 == 0):
                src = os.path.join(input_dir, filename)
                dst = os.path.join(output_dir, filename)
                shutil.copy2(src, dst)

def main() -> None:
    """
    Main function to handle user input and orchestrate the copying of files.
    """
    print("scripts.py is just for creating subsets of datasets at the moment")
    input_dir = prompt_for_directory("Select Input Directory")
    if not input_dir:
        print("No input directory selected. Exiting.")
        return
    
    output_dir = prompt_for_directory("Select Output Directory")
    if not output_dir:
        print("No output directory selected. Exiting.")
        return
    
    try:
        end_number = int(input("Enter the ending number: "))
    except ValueError:
        print("Invalid number entered. Exiting.")
        return
    
    even_only = False
    even_only_raw = input("Copy even numbers only? (y/n): ")
    if even_only_raw.lower().strip() == "y":
        even_only = True
        end_number = end_number if end_number % 2 == 0 else end_number - 1

    print("Copying files...")
    copy_files(input_dir, output_dir, end_number, even_only)
    print("Done!")

if __name__ == "__main__":
    main()
