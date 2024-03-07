import torch
from torchvision import transforms
from PIL import Image
from pytorchColorModel import ColorModel  # Import the necessary modules from your training script

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

def load_model(model_path, num_classes):
    model = ColorModel(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        shape_output, letter_output = model(image_tensor)

    _, predicted_shape = torch.max(shape_output, 1)
    _, predicted_letter = torch.max(letter_output, 1)

    predicted_shape_color = LABEL_TO_COLOR_DICT[predicted_shape.item()]
    predicted_letter_color = LABEL_TO_COLOR_DICT[predicted_letter.item()]

    print("Predicted Shape Color:", predicted_shape_color)
    print("Predicted Letter Color:", predicted_letter_color)

if __name__ == "__main__":
    # Replace 'trained_model.pth' and num_classes with your actual model path and number of classes
    model_path = 'trained_model.pth'
    num_classes = 8

    # Load the trained model
    model = load_model(model_path, num_classes)

    # Replace 'path/to/unlabeled/image.jpg' with the path to your unlabeled image
    image_path = './test/dataset/data/0.jpg'

    # Make predictions on the unlabeled image
    predict_image(model, image_path)
