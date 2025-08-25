#MIA_target_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from attackModel_MIA import LeNet
import numpy as np

def train_target_model():
    # Data preparation: Normalize the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', download=True, transform=transform)
    
    # Split dataset according to the scenario train_size and test_Size
    train_size = int(0.5 * len(dataset)) # Apply more train size to mitigate overfitting
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create data loaders for training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device) # Learning rate for training the target model
    # optimizer = optim.Adam(model.parameters(), lr=0.001) # loss function for multi class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6) # L2 Regularization

    criterion = nn.CrossEntropyLoss()

    # Training the target model
    model.train()
    for epoch in range(50): # Number of epochs for training
        model.train()
        correct_train = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            correct_train += pred.eq(target.view_as(pred)).sum().item()

        train_accuracy = 100. * correct_train / len(train_loader.dataset)

        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct_test = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct_test += pred.eq(target.view_as(pred)).sum().item()

        test_accuracy = 100. * correct_test / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)

        # Print training and test accuracy for each epoch
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
    
    # Save the target model, scenario overfit and well-trained
    # Overfitted target model (small training set)
    # torch.save(model.state_dict(), 'target_model_overfitted.pth') 
    # Well trained target model (larger training set)
    torch.save(model.state_dict(), 'target_model_well_trained.pth')


if __name__ == '__main__':
    train_target_model()
