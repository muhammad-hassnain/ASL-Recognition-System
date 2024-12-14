import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DenseNet121ConvNet(nn.Module):
    def __init__(self, num_classes=26, pretrained=True):
        super(DenseNet121ConvNet, self).__init__()
        
        # Load pre-trained DenseNet121 with default weights
        self.densenet = models.densenet121(pretrained=pretrained)
        
        # Freeze all layers in DenseNet121 (optional)
        for param in self.densenet.parameters():
            param.requires_grad = False
        
        # Custom classification head
        # Get the number of input features from DenseNet121 after the convolutional layers
        feature_dim = self.densenet.classifier.in_features
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Resize the input to 224x224 (DenseNet input size)
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        
        # Extract features using DenseNet121
        x = self.densenet.features(x)
        
        # Global Average Pooling (reduce spatial dimensions to a single value per channel)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # Flatten the tensor to pass it into the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return self.softmax(x)