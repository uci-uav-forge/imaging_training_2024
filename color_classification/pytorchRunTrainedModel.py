import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from pytorchColorModel import ColorModel

LABEL_TO_COLOR_DICT = {
    0: "red",
    1: "orange",
    2: "yellow",
    3: "green",
    4: "blue",
    5: "purple",
    6: "white",
    7: "black",
    8: "brown",
    9: "gray",
}

# change test dataset directory here
test_directory = './test/dataset'
test_df = pd.read_csv(test_directory + '/labels.txt')

def test_read_image(image_file):
    image = Image.open(test_directory + image_file)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    return image

test_file_paths = test_df['file'].values
dataset = (test_read_image(file) for file in test_file_paths)

model = ColorModel(num_classes=8)  # Assuming num_classes is the same as during training
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

total_correct_shape_color = 0
total_correct_letter_color = 0
total_samples = 0

for index, image in enumerate(dataset):
    with torch.no_grad():
        shape_output, letter_output = model(image)
    
    _, predicted_shape = torch.max(shape_output, 1)
    _, predicted_letter = torch.max(letter_output, 1)

    # Get corresponding labels from test_df
    shape_colors = test_df[' shape_color'].values[index]
    letter_colors = test_df[' letter_color'].values[index]

    total_correct_shape_color += (predicted_shape == shape_colors).sum().item()
    total_correct_letter_color += (predicted_letter == letter_colors).sum().item()
    total_samples += 1

shape_accuracy = total_correct_shape_color / total_samples
letter_accuracy = total_correct_letter_color / total_samples

print(f"Shape Accuracy: {shape_accuracy:.4f}")
print(f"Letter Accuracy: {letter_accuracy:.4f}")
print(f"Total {total_samples}")

