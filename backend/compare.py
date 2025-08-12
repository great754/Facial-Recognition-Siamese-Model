## dataset prep
from time import time

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


##TODO
# Feature extractor based on Classifier
class FeatureExtractor(nn.Module):
    def __init__(self):
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

