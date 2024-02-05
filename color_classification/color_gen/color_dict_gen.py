import os

def build_colors_dict(root_folder):
    colors_dict = {}

    for color_folder in os.listdir(root_folder):
        color_path = os.path.join(root_folder, color_folder)

        if os.path.isdir(color_path):
            rgb_list = []

            for filename in os.listdir(color_path):
                if filename.endswith(".jpg"):
                    rgb_values = tuple(map(int, filename[1:-5].split(',')))
                    rgb_list.append(rgb_values)

            colors_dict[color_folder] = rgb_list

    return colors_dict

def write_dict_to_file(colors_dict, output_file):
    with open(output_file, 'w') as f:
        f.write(f"colors_dict = {colors_dict}")

if __name__ == "__main__":
    root_folder = "./colors"  
    output_file = "colors_output.py" 

    colors_dict = build_colors_dict(root_folder)
    write_dict_to_file(colors_dict, output_file)
