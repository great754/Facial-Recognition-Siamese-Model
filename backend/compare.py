## dataset prep
from time import time
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import matplotlib.pyplot as plt
from scipy.stats import loguniform

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from torch.utils.data import Dataset, DataLoader
import numpy as np

import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Assuming X and y are already defined as in the provided code
# fig, axes = plt.subplots(2, 5, figsize=(15, 8))

# for i, ax in enumerate(axes.flat):
#     ax.imshow(X[i])
#     ax.set_title(f"Label: {target_names[y[i]]}")
#     ax.axis('off')  # Hide axis ticks and labels

# plt.tight_layout()
# plt.show()

## contrastive loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        #TODO
        super(ContrastiveLoss, self).__init__()
        self.margin = margin    ## the threshold that checks how similar or dissimilar they are
    def forward(self, distance, label):
      sim_loss = 0.5 * (1-label) * torch.pow(torch.abs(distance), 2)
      ## dis_loss = 0.5 * label * max(self.margin - distance, 0.0) can't use this since max is a pytorch tensor
      dis_loss = 0.5 * label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)   ## clamp gets the max
      loss = torch.mean(sim_loss + dis_loss)    ## get the average loss over the batch since we do batches
      return loss

# Feature extractor based on Classifier
class FeatureExtractor(nn.Module):
    def __init__(self, n_features=37*50*3):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(n_features, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 32)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

# Siamese Network
class SiameseNetwork_dist_withoutCNN(nn.Module):
    def __init__(self):
        super(SiameseNetwork_dist_withoutCNN, self).__init__()
        self.feature_extractor = FeatureExtractor()
        #self.fc_compare = nn.Linear(32, 1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        f1 = self.feature_extractor(x1)
        f2 = self.feature_extractor(x2)
        ##distance = torch.abs(f1 - f2)  # L1 distance
        ##out = self.fc_compare(distance)
        ##return self.sigmoid(out)
        return torch.norm(f1 - f2, p=2, dim=1)     ## so we just get euclidean distance of the embeddings directly, and keep the dimensions for every item in the batch


model = SiameseNetwork_dist_withoutCNN()

path = 'backend/best_model.pth'
state_dict = torch.load(path)
model.load_state_dict(state_dict)

model.eval()

def same_person(img1, img2):
    """
    Selects two examples from X, preprocesses them, and runs them through the model.
    idx1, idx2: indices of the examples to use (default: 0 and 1)
    """
    # Flatten and convert to torch tensors
    x1 = torch.tensor(img1.reshape(-1), dtype=torch.float32).unsqueeze(0)
    x2 = torch.tensor(img2.reshape(-1), dtype=torch.float32).unsqueeze(0)
    # Run through model
    out = model(x1, x2)
    # print(out)
    return out
# x1 = torch.tensor(X[0].reshape(-1), dtype=torch.float32).unsqueeze(0)
# x2 = torch.tensor(X[3].reshape(-1), dtype=torch.float32).unsqueeze(0)


def find_two_same_person_indices(y):
    indices = []
    seen = set()
    
    for i in range(len(y)):
        if len(indices) >= 10:
            return indices
        for j in range(i+1, len(y)):
            if y[i] == y[j]:
                if y[i] not in seen:
                    indices.append((i, j))
                    seen.add(y[i])
    return indices

# indices = find_two_same_person_indices(y)
# idx1, idx2 = indices[8]
# print(f"Indices of same person: {idx1}, {idx2} (Label: {target_names[y[idx1]]})")
# print(same_person(X[idx1], X[2]))

# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# axes[0].imshow(X[idx1])
# axes[0].set_title(f"X1 (Label: {target_names[y[idx1]]})")
# axes[0].axis('off')
# axes[1].imshow(X[2])
# axes[1].set_title(f"X2 (Label: {target_names[y[2]]})")
# axes[1].axis('off')
# plt.tight_layout()
# plt.show()