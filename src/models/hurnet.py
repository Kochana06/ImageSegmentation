import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class HURNet(nn.Module):
    def __init__(self, num_classes=3):
        super(HURNet, self).__init__()
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])  # Remove FC layers

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),  
            nn.Sigmoid()
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        mask = self.decoder(features)  
        mask = F.interpolate(mask, size=(224, 224), mode="bilinear", align_corners=False)  # Ensure 224x224 size
        
        pooled_features = self.global_avg_pool(features)  
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  
        classification = self.classifier(pooled_features)

        return mask, classification
    

