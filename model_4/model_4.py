import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MobileNetV2ConvNet(nn.Module):
    def __init__(self, num_classes=26, pretrained=True):
        super(MobileNetV2ConvNet, self).__init__()
        
        # Load pre-trained MobileNetV2 with default weights
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Freeze all layers in MobileNetV2 (optional)
        for param in self.mobilenet.parameters():
            param.requires_grad = False
        
        # Custom classification head
        # Get the number of input features from MobileNetV2 after the convolutional layers
        feature_dim = self.mobilenet.classifier[1].in_features
        
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
        # Resize the input to 224x224 (MobileNetV2 input size)
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        
        # Extract features using MobileNetV2
        x = self.mobilenet.features(x)
        
        # Global Average Pooling (reduce spatial dimensions to a single value per channel)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # Flatten the tensor to pass it into the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return self.softmax(x)
