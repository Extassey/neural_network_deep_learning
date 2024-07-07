import os
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# Set Device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define the Model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)  # Initialize the model without pre-trained weights
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # We have 3 classes: dog, cat, and horse
model = model.to(device)

# Load the saved model weights
model.load_state_dict(torch.load(r'c:\Users\darkl\Desktop\saved_model\best_model.pth'))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the class names
class_names = ['cat', 'dog', 'horse']

def predict_image(image_path):
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make the prediction
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds.item()]

    return predicted_class

# Main script
if __name__ == "__main__":
    image_path = input("Enter the path to the image: ")
    if os.path.exists(image_path):
        prediction = predict_image(image_path)
        print(f"The model predicts that the image is a: {prediction}")
    else:
        print("The specified path does not exist.")
