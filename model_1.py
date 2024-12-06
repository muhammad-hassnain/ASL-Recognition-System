import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 9)
        self.bn2 = nn.BatchNorm2d(16)
        # Placeholder for fully connected layer size
        self.fc2 = nn.Linear(16 * 16 * 16, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 26)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # print(x.shape)  
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return self.softmax(x)
