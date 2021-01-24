import torch
import torch.nn as nn
import torchvision
from timm.models.efficientnet import tf_efficientnet_b5_ns

class Detector(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(Detector, self).__init__()
        # params 
        self.feature_extractor = tf_efficientnet_b5_ns(pretrained=True, drop_path_rate=0.2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(2048,256)
        self.fc2 = nn.Linear(256, 11)
  
    def forward(self, image):
        # get feature
        feature = self.feature_extractor.forward_features(image) 
        feature = self.avg_pool(feature).flatten(1)
        feature = self.dropout(feature)

        # do fully connected layers
        x = self.fc1(feature)
        x = self.fc2(x)

        return x