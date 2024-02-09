import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import DataLoader
from pytorchColorModel import ColorModel, ColorDataset  # Import the necessary modules from your training script

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

# Assuming num_classes is the same as during training
num_classes = 8

def test_read_image(image_file):
    image = Image.open(image_file)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    return image

def evaluate_colors(model, test_loader, test_df):
    model.eval()

    total_correct_shape_color = 0
    total_correct_letter_color = 0
    total_samples = 0

    # Create a dictionary to store accuracies
    accuracies = defaultdict(list)

    for index, (image, letter_color, shape_color) in enumerate(test_loader):
        letter_color = letter_color.numpy()[0]
        shape_color = shape_color.numpy()[0]
        with torch.no_grad():
            letter_output, shape_output = model(image)

        _, predicted_shape = torch.max(shape_output, 1)
        _, predicted_letter = torch.max(letter_output, 1)
        

        total_correct_shape_color += int(predicted_shape == shape_color)
        total_correct_letter_color +=  int(predicted_letter == letter_color)
        total_samples += 1
        print(total_correct_letter_color, total_samples)

        # Calculate accuracies for each shape-letter color combination

        if len(accuracies[f"{LABEL_TO_COLOR_DICT[shape_color]}_{LABEL_TO_COLOR_DICT[letter_color]}"]) == 0:
            accuracies[f"{LABEL_TO_COLOR_DICT[shape_color]}_{LABEL_TO_COLOR_DICT[letter_color]}"] = [0, 0, 0] # [L_correct, s_correct, total]

        # Calculate shape color accuracy
        shape_color_correct = (predicted_shape == shape_color)
        # Calculate letter color accuracy
        letter_color_correct = (predicted_letter == letter_color)
        if letter_color_correct:
            accuracies[f"{LABEL_TO_COLOR_DICT[shape_color]}_{LABEL_TO_COLOR_DICT[letter_color]}"][0] += 1
        if shape_color_correct:
            accuracies[f"{LABEL_TO_COLOR_DICT[shape_color]}_{LABEL_TO_COLOR_DICT[letter_color]}"][1] += 1
        accuracies[f"{LABEL_TO_COLOR_DICT[shape_color]}_{LABEL_TO_COLOR_DICT[letter_color]}"][2] += 1

            
    # Print and return the accuracies
    for key, value in accuracies.items():
        letter_accuracy = value[0] / value[2]
        
        shape_accuracy = value[1] / value[2]
        print(f"{key} Accuracy:", letter_accuracy, shape_accuracy)

    shape_accuracy = total_correct_shape_color / total_samples
    letter_accuracy = total_correct_letter_color / total_samples

    print(f"Shape Accuracy: {shape_accuracy:.4f}")
    print(f"Letter Accuracy: {letter_accuracy:.4f}")

    return accuracies

if __name__ == "__main__":
    # Load the trained model
    model = ColorModel(num_classes)
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()

    # Load the test dataset
    test_directory = './test/dataset'
    test_df = pd.read_csv(test_directory + '/labels.txt')
    test_file_paths = test_df['file'].apply(lambda x: test_directory + x).values

    # Create a DataLoader for the test dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    test_dataset = ColorDataset(test_file_paths, test_df[' letter_color'].values, test_df[' shape_color'].values, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluate the model on the test dataset
    evaluate_colors(model, test_loader, test_df)
