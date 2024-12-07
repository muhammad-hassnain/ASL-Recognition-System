import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet50ConvNet(nn.Module):
    def __init__(self, num_classes=26, pretrained=True):
        super(ResNet50ConvNet, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the original classification layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Freeze feature extraction layers (optional)
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Custom classification head
        # Determine the number of features from ResNet50
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 224, 224)
            feature_dim = self.features(sample_input).view(1, -1).size(1)
        
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
        # Upsample input to 224x224 (ResNet50 input size)
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        
        # Extract features using ResNet50
        x = self.features(x)
        
        # Flatten the features
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return self.softmax(x)