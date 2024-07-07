import torch
from torch import nn
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the Neural Network class (same as the one used for training)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Load the model weights (specify the full path to your weights file)
model = NeuralNetwork()
model.load_state_dict(torch.load(r'c:\Users\darkl\Downloads\neural-networks-and-deep-learning-master\neural-networks-and-deep-learning-master\model_weights.pth'))
model.eval()  # Set the model to evaluation mode

# Define the transformation to convert image to tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# Define the class names for Fashion-MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to make a prediction
def predict(model, image):
    with torch.no_grad():
        image = transform(image).unsqueeze(0)  # Preprocess the image and add batch dimension
        output = model(image)
        probabilities = nn.Softmax(dim=1)(output)
        predicted_class = probabilities.argmax(1).item()
    return predicted_class

# Function to load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path)
    return image

# Function to display the image and prediction
def display_prediction(image_path):
    image = load_image(image_path)
    predicted_class = predict(model, image)
    
    plt.imshow(image, cmap='gray')
    plt.title(f'Predicted: {class_names[predicted_class]}')
    plt.show()

# Example usage: display prediction for a custom image
# Replace 'your_image_path_here.png' with the path to your image file
display_prediction(r'c:\Users\darkl\Pictures\sneaker_3.jpg')
