import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

class ColorDataset(Dataset):
    def __init__(self, file_paths, letter_colors, shape_colors, transform=None):
        self.file_paths = file_paths
        self.letter_colors = letter_colors
        self.shape_colors = shape_colors
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx])
        img = self.transform(img)
        letter_color = torch.tensor(self.letter_colors[idx], dtype=torch.long)
        shape_color = torch.tensor(self.shape_colors[idx], dtype=torch.long)
        return img, letter_color, shape_color


class ColorModel(nn.Module):
    def __init__(self, num_classes):
        super(ColorModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.letter_dense = nn.Sequential(
            nn.Linear(128 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )

        self.shape_dense = nn.Sequential(
            nn.Linear(128 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)

        letter_output = self.letter_dense(x)
        shape_output = self.shape_dense(x)
        
        return letter_output, shape_output

if __name__ == "__main__":
    num_classes = 8
    learning_rate = 0.0001
    batch_size = 24
    epochs = 3

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_directory = './train/dataset'
    train_df = pd.read_csv(train_directory + '/labels.txt')
    train_file_paths = train_df['file'].apply(lambda x: train_directory + x).values
    train_letter_colors = train_df[' letter_color'].values
    train_shape_colors = train_df[' shape_color'].values

    # Create dataset and dataloader
    dataset = ColorDataset(train_file_paths, train_letter_colors, train_shape_colors, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = ColorModel(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_shape_correct = 0
        total_letter_correct = 0
        total_samples = 0

        for image, letter_colors, shape_colors in dataloader:
            optimizer.zero_grad()
            outputs = model(image)
            print(torch.max(outputs[0], 1)[1], letter_colors)
            letter_loss = criterion(outputs[0], letter_colors)
            shape_loss = criterion(outputs[1], shape_colors)
            # print(shape_loss)
            total_loss = letter_loss + shape_loss # this seems whack
            total_loss.backward()
            
            optimizer.step()

            # Calculate accuracies
            _, predicted_letter = torch.max(outputs[0], 1)
            _, predicted_shape = torch.max(outputs[1], 1)
            # print(f"predicted {predicted_shape} {shape_colors}")
            total_shape_correct += (predicted_shape == shape_colors).sum().item()
            total_letter_correct += (predicted_letter == letter_colors).sum().item()
            total_samples += letter_colors.size(0)
            

        # Calculate accuracies for the epoch
        shape_accuracy = total_shape_correct / total_samples
        letter_accuracy = total_letter_correct / total_samples

        print(f"Epoch {epoch + 1}/{epochs}: ",
              f"Shape Accuracy: {shape_accuracy:.4f}, "
              f"Letter Accuracy: {letter_accuracy:.4f}")

    # Save the model
    torch.save(model.state_dict(), "./trained_model.pth")
