#first program dependencies

import os
import torch
import time
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

#added later dependencies
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

#initialization logic
userInput = input("type in train to train, or prediction to predict: ")
if (userInput == "train" or "Train"):
    # Set Device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Prepare the Dataset
    data_dir = r'c:\Users\darkl\Desktop\dataset'

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Define the Model
    model = resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)  # We have 3 classes: dog and cat and horse
    model = model.to(device)

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Directory to save the model
    save_dir = r'c:\Users\darkl\Desktop\saved_model'
    os.makedirs(save_dir, exist_ok=True)

    # Train the Model
    def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=5):
        best_model_wts = model.state_dict()
        best_acc = 0.0

        for epoch in range(num_epochs):
            start_time = time.time()
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            # Training phase
            model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)

            print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Validation phase
            model.eval()
            running_loss = 0.0
            running_corrects = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(val_loader.dataset)
            epoch_acc = running_corrects.double() / len(val_loader.dataset)

            print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the model if it has the best accuracy so far
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

            end_time = time.time()
            print(f"Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds")

        # Load best model weights
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        print(f"Best model saved with accuracy: {best_acc:.4f}")

        return model

    # Run the training process
    trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=5)
elif (userInput == "predict"):
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
else: 
    print("error")