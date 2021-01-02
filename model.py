import torch
import torchvision
from efficientnet_pytorch import EfficientNet


class Detector(torch.nn.Module):
    def __init__(self, image_size, model_name="efficientnet-b4"):
        super(Detector, self).__init__()
        # params 
        self.image_size = image_size
        self.feature_extractor = EfficientNet.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(1792,256)
        self.fc2 = torch.nn.Linear(256, 11)
  
    def forward(self, image):
        # get feature
        feature = self.feature_extractor.extract_features(image) # B, 1792, 16, 16 (B4)
        feature = torch.nn.AvgPool2d((feature.shape[-2],feature.shape[-1]))(feature)
        feature = feature.squeeze() # B, 1792
        feature = self.dropout(feature)

        # do fully connected layers
        x = self.fc1(feature)
        x = self.fc2(x)

        return x