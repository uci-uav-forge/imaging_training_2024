import os
import shutil
import re
from tkinter import Tk, filedialog

# Function to open a folder dialog
def select_directory(title):
    root = Tk()
    root.withdraw()  # Hide the main window
    directory = filedialog.askdirectory(title=title)
    root.destroy()
    return directory

# Function to copy files
def copy_files(input_dir, output_dir, end_number, even_only=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Adjusted regex pattern to match the number at the end after the last underscore
    pattern = re.compile(r'.*_([0-9]+)\.')
    
    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            file_number = int(match.group(1))
            if file_number <= end_number and (not even_only or file_number % 2 == 0):
                src = os.path.join(input_dir, filename)
                dst = os.path.join(output_dir, filename)
                shutil.copy2(src, dst)

# Main function
def main():
    print("scripts.py is just for data splitting atm")
    input_dir = select_directory("Select Input Directory")
    if not input_dir:
        print("No input directory selected. Exiting.")
        return
    
    output_dir = select_directory("Select Output Directory")
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
