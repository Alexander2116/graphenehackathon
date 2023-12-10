import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import random
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class LaserRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LaserRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        #self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        #x = self.sigmoid(x)
        return x


input_size = 247
hidden_size = 128
output_size = 3

weigths_path = '.\\499.torch'

# Instantiate the model
model = LaserRegression(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(weigths_path))
#model = LaserCNN(input_channels,hidden_channels,output_channels,conv_kernel_size,pool_kernel_size,hidden_size,output_size)
model.to(device)
model.eval()

data_tensor = torch.tensor(data, dtype=torch.float32)

with torch.no_grad():
    outputs = model(data_tensor)
    probabilities = F.softmax(outputs, dim=1)
    _, predicted_classes = probabilities.max(dim=1)
    print('Predicted class',predicted_classes)