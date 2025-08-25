#MIA_attack_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from attackModel_MIA import LeNet, AttackModel
from torchvision import datasets, transforms
import time

def train_attack_model():
    #start timing
    start_time = time.time()
    
    # Data preparation: normalize the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', download=True, transform=transform)
    
    # split dataset into the desired scenario
    train_size = int(0.5 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create data loaders for training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model = LeNet().to(device)
    
    # Load the target model and set it to evaluation mode
    # target_model.load_state_dict(torch.load('target_model_overfitted.pth'))
    target_model.load_state_dict(torch.load('target_model_well_trained.pth'))
    target_model.eval()
    
    # Initialize attack model
    attack_model = AttackModel().to(device)
    # low learning rate for better converge
    optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
     # Binary cross entropy for binary classification
    criterion = nn.BCELoss()

    # Preparing attack dataset by collecting outputs from target model
    attack_data = []
    attack_labels = []

   # Collect outputs for member samples (training set)
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            output = target_model(data)
            attack_data.append(output.cpu().numpy())
            attack_labels.append(np.ones(len(target))) # label as 1 for members
        
        # Collect outputs for non - member samples (training set)
        for data, target in test_loader:
            data = data.to(device)
            output = target_model(data)
            attack_data.append(output.cpu().numpy())
            attack_labels.append(np.zeros(len(target))) # label as 0 for non-member

    # combine member and non-member outputs and labels
    attack_data = np.concatenate(attack_data)
    attack_labels = np.concatenate(attack_labels)

    # Ensure dataset is balanced
    # Balance the attack dataset by sampling an equal number of non-member samples
    pos_indices = np.where(attack_labels == 1)[0]
    neg_indices = np.where(attack_labels == 0)[0]
    sampled_neg_indices = np.random.choice(neg_indices, len(pos_indices), replace=True)  # Use replace=True
    balanced_indices = np.concatenate([pos_indices, sampled_neg_indices])
    
    attack_data = attack_data[balanced_indices]
    attack_labels = attack_labels[balanced_indices]

    # Convert numpy arrays to PyTorch tensors
    attack_data = torch.tensor(attack_data, dtype=torch.float32).to(device)
    attack_labels = torch.tensor(attack_labels, dtype=torch.float32).to(device)

    # Training the attack model
    attack_model.train()
    for epoch in range(250):  # Increase the number of epochs for training
        optimizer.zero_grad()
        output = attack_model(attack_data)
        loss = criterion(output.squeeze(), attack_labels)
        loss.backward()
        optimizer.step()
        print(f'Attack Model Epoch {epoch+1}, Loss: {loss.item()}')

    # Save the attack model
    # torch.save(attack_model.state_dict(), 'attack_model.pth')

    # Evaluate the attack model
    attack_model.eval()
    with torch.no_grad():
        attack_predictions = attack_model(attack_data).squeeze().cpu().numpy()
        attack_predictions = np.round(attack_predictions)  # Convert probabilities to binary predictions

    # Calculate evaluation metrics
    accuracy = accuracy_score(attack_labels.cpu().numpy(), attack_predictions)
    precision = precision_score(attack_labels.cpu().numpy(), attack_predictions, zero_division=1)
    recall = recall_score(attack_labels.cpu().numpy(), attack_predictions, zero_division=1)
    f1 = f1_score(attack_labels.cpu().numpy(), attack_predictions, zero_division=1)
    
    end_time = time.time()
    total_time = end_time - start_time

    # Print the evaluation metrics
    print(f'Attack Model Evaluation:\n'
          f'Accuracy of the attack model on the test images: {accuracy:.4f}\n'
          f'MIA Attack Time taken: {total_time} seconds\n'
          f'Precision: {precision:.4f}\n'
          f'Recall: {recall:.4f}\n'
          f'F1 Score: {f1:.4f}')

if __name__ == '__main__':
    train_attack_model()
