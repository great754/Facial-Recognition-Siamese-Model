import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
# Load pretrained ResNet (remove classifier head)
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        # Remove the classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer

    def forward(self, x):
        x = self.backbone(x)  # Shape: (batch, 512, 1, 1)
        x = torch.flatten(x, 1)  # Shape: (batch, 512)
        return x

# Siamese network with ResNet backbone
class SiameseNetwork_withResNet(nn.Module):
    def __init__(self):
        super(SiameseNetwork_withResNet, self).__init__()
        self.feature_extractor = ResNetFeatureExtractor()

    def forward(self, x1, x2):
        f1 = self.feature_extractor(x1)
        f2 = self.feature_extractor(x2)
        return torch.norm(f1 - f2, p=2, dim=1)  # Euclidean distance

# Load your trained Siamese model state dict
model = SiameseNetwork_withResNet()
# model.load_state_dict(torch.load('backend/best_model.pth', map_location='cpu'))
model.eval()

# Transform to match ResNet requirements
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet default
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ResNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

def same_person(img1, img2):
    """
    img1, img2: NumPy arrays (H, W, C) or PIL Images.
    """
    # Convert to PIL if needed
    if isinstance(img1, np.ndarray):
        img1 = Image.fromarray((img1 * 255).astype(np.uint8)) if img1.max() <= 1 else Image.fromarray(img1.astype(np.uint8))
    if isinstance(img2, np.ndarray):
        img2 = Image.fromarray((img2 * 255).astype(np.uint8)) if img2.max() <= 1 else Image.fromarray(img2.astype(np.uint8))

    # Apply preprocessing
    x1 = transform(img1).unsqueeze(0)
    x2 = transform(img2).unsqueeze(0)

    with torch.no_grad():
        distance = model(x1, x2)

    # print(f"Euclidean distance: {distance.item()}")
    return distance