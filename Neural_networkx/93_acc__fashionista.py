import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the ImprovedNeuralNetwork class (same as your training script)
class ImprovedNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Load the trained model weights
model = ImprovedNeuralNetwork()
model.load_state_dict(torch.load(r'c:\Users\darkl\Downloads\neural-networks-and-deep-learning-master\neural-networks-and-deep-learning-master\model_weights.pth'))
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
    transforms.Resize((28, 28)),  # Resize the image to 28x28 pixels
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image
])

def predict(image_path):
    # Load and transform the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Display the image for debugging
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title("Input Image")
    plt.show()

    # Make a prediction
    with torch.no_grad():
        output = model(image)
        predicted_label = output.argmax(1).item()

    return predicted_label

# Example usage
image_path = r'c:\Users\darkl\Downloads\sneaker.jpg'  # Replace with your image path
predicted_label = predict(image_path)
print(f'Predicted label: {predicted_label}')
