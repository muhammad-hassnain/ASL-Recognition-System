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




# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 3)  # Change 1 to 3 for RGB images
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 4)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 26)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return self.softmax(x)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # Convolutional Layers
#         self.conv1 = nn.Conv2d(3, 6, 3)  # RGB input
#         self.bn1 = nn.BatchNorm2d(6)  # Batch Normalization
#         self.conv2 = nn.Conv2d(6, 16, 4)
#         self.bn2 = nn.BatchNorm2d(16)
        
#         # Pooling Layer
#         self.pool = nn.AdaptiveAvgPool2d((5, 5))  # Adaptive Pooling

#         # Fully Connected Layers
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 26)
        
#         # Dropout for regularization
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         # Convolutional Layers with BatchNorm and Activation
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.pool(x)
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool(x)

#         # Flattening
#         x = x.view(-1, 16 * 5 * 5)

#         # Fully Connected Layers with Dropout
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)  # Raw logits
#         return x  # Softmax applied externally in the loss function

