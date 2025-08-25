#attackModel_MIA.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(6, 16, 5, stride = 1, padding = 0)
        self.dropout1 = nn.Dropout(0.5) # Apply dropout layer
        self.dropout2 = nn.Dropout(0.5) # Apply second dropout layer
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x) # apply dropout
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout2(x) # apply dropout
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x) # apply dropout again
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1) # apply dropout again
    
        return output

# Define the Attack Model as a fully connected network
# Define the Attack Model as a more complex fully connected network
class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(10, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x