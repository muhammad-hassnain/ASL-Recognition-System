import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AlexNetConvNet(nn.Module):
    def __init__(self, num_classes=26, pretrained=True):
        super(AlexNetConvNet, self).__init__()
        
        # Load pre-trained AlexNet
        self.alexnet = models.alexnet(pretrained=pretrained)
        
        # Remove the original classification layer
        self.features = self.alexnet.features
        
        # Freeze feature extraction layers (optional)
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Determine the number of input features from AlexNet's last layer
        feature_dim = self.alexnet.classifier[1].in_features
        
        # Custom classification layers
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
        # Upsample input to 224x224 (AlexNet input size)
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        
        # Extract features using AlexNet
        x = self.features(x)
        
        # Global Average Pooling (reduce spatial dimensions to a single value per channel)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return self.softmax(x)
