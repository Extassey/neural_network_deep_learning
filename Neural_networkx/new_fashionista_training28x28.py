import os
import torch
from torch import nn
from torchvision import transforms
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

# Define the Model (assuming this matches your saved model architecture)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = SimpleNN().to(device)

# Load the saved model weights
model.load_state_dict(torch.load(r'c:\Users\darkl\Downloads\neural-networks-and-deep-learning-master\neural-networks-and-deep-learning-master\model_weights.pth'))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize the image to 28x28 pixels
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize the image
])

# Define the class names for Fashion-MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def predict_image(image_path):
    # Load and transform the image
    image = Image.open(image_path).convert('L')  # Open image in grayscale mode
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
