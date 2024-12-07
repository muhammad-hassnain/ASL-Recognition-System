import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG16ConvNet(nn.Module):
    def __init__(self, num_classes=26, pretrained=True):
        super(VGG16ConvNet, self).__init__()
        
        # Load pre-trained VGG16
        self.vgg = models.vgg16(pretrained=pretrained)
        
        # Remove the original classification layer
        self.features = nn.Sequential(*list(self.vgg.features))
        
        # Freeze feature extraction layers (optional)
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Custom classification head
        # Determine the number of features from VGG16
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 224, 224)
            feature_dim = self.vgg.classifier[0].in_features
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(1024, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Upsample input to 224x224 (VGG16 input size)
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        
        # Extract features using VGG16
        x = self.features(x)
        
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # Flatten the features
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return self.softmax(x)