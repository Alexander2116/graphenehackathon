import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import random
import wandb
import torch.nn.functional as F

learning_rate = 0.001
num_epochs =  500

wandb.init(
    # Set the project where this run will be logged
    project="graphene_hackathon_fnn",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "conv_kernel": 3,
        "conv_layers": 2,
        "hidden_channels":8,
        "output_channels":16,
        "pool_kernel":2,
    },
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data.iloc[idx, 1:].values, dtype=torch.float32)
        
        if self.transform:
            sample = self.transform(sample)

        label = torch.tensor(self.data.iloc[idx, 0], dtype=torch.float32)
        return sample, label



class MultiClassDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        self.data, self.labels = self.load_data()

    def load_data(self):
        all_data = np.zeros((247))
        all_labels = np.zeros((247))
        #all_labels = np.zeros((1))
        label = 0
        for file_path in self.file_paths:
            class_data = np.genfromtxt(file_path,delimiter=',')
            mean_data = np.mean(class_data)
            class_data -= mean_data
            all_data = np.vstack((all_data,class_data))
            filename = file_path.replace(working_directory,"")
            letter = filename.replace('train/',"")[0]
            if letter == 'c':
                label = 0
            else:
                letter = filename.replace('train/',"")[6]
                if letter == 'b':
                    label = 2
                else:
                    label = 1
            label = np.ones(247)*label
            #label = np.array(label)
            all_labels = np.vstack((all_labels,label)).astype(int)
            #all_labels = np.vstack((all_labels,label)).astype(int)
        all_data = all_data[1::]
        all_labels = all_labels[1::]
        return all_data, all_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #sample = torch.tensor(self.data.iloc[idx, 1:].values[0], dtype=torch.float32)
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)

        #label = torch.tensor(self.labels[idx][0],dtype=torch.long)
        label = torch.tensor(self.labels[idx][0],dtype=torch.long)
        return sample, label


"""
class MultiClassDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        self.data, self.labels = self.load_data()

    def load_data(self):
        all_data = np.zeros((1,10,500))
        all_labels = np.zeros((1))
        for file_path in self.file_paths:
            class_data = np.load(file_path)
            class_data.shape = [1,10,500]
            all_data = np.vstack((all_data,class_data))
            filename = file_path.replace(working_directory,"")
            if filename[1]=='r':
                letter = filename.replace('train/',"")[0]
                if letter == 'l':
                    label = 0
                else:
                    label = 1
            else:
                letter = filename.replace('test/',"")[0]
                if letter == 'l':
                    label = 0
                else:
                    label = 1
            all_labels = np.vstack((all_labels,label))
        all_data = all_data[1::]
        all_labels = all_labels[1::]
        return all_data,all_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #sample = torch.tensor(self.data.iloc[idx, 1:].values[0], dtype=torch.float32)
        sample = self.data[idx]
        sample.shape = [1,10,500]
        if self.transform:
            sample = self.transform(sample)

        #label = torch.tensor(int(self.data[idx, 0]), dtype=torch.float32).reshape(-1, 1)  # Assuming class labels are integers
        label = torch.tensor(self.labels[idx],dtype=torch.float32)
        return sample, label
"""

# Define a simple transform to tensor
class ToTensor:
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)


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

"""
class LaserCNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, conv_kernel_size, pool_kernel_size, hidden_size, output_size):
        super(LaserCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, conv_kernel_size)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels,output_channels,conv_kernel_size)
        self.batchnorm2 = nn.BatchNorm2d(output_channels)
        self.maxpool = nn.MaxPool2d(pool_kernel_size,stride=2)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x

"""

"""
# Define input, hidden, and output sizes
num_batch = 2
input_size = 500
input_channels = 1
hidden_channels = 8
output_channels = 16
conv_kernel_size = 3
pool_kernel_size = 2
hidden_size = 248*3*output_channels
output_size = 1
"""

input_size = 247
hidden_size = 128
output_size = 3
num_batch = 2

working_directory = 'Documents/graphene_hackathon/real_data/'

# Instantiate the model
model = LaserRegression(input_size, hidden_size, output_size)
#model = LaserCNN(input_channels,hidden_channels,output_channels,conv_kernel_size,pool_kernel_size,hidden_size,output_size)
model.to(device)
#csv_file_path = 'train_set.csv'
#train_dataset = CustomDataset(csv_file_path, transform=ToTensor())
train_file_paths = glob.glob(working_directory+'train/*Al*.csv')
random.shuffle(train_file_paths)

train_dataset = MultiClassDataset(train_file_paths, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=num_batch, shuffle=False)

test_file_paths = glob.glob(working_directory+'train/*Al*.csv')
random.shuffle(test_file_paths)
test_dataset = MultiClassDataset(test_file_paths, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
model.train()

print('Starting training')
for epoch in range(num_epochs):
    model.train()
    batch_loss_array = []
    print('Epoch',epoch)
    for batch_index, batch in enumerate(train_loader):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        batch_loss = loss(outputs, labels)
        batch_loss_array.append(batch_loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    # Print training progress
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {sum(batch_loss_array)/batch_index}')
    wandb.log({"loss": sum(batch_loss_array)/batch_index})
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, targets in test_loader:
            # Forward pass
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted_classes = probabilities.max(dim=1)
            #predicted = torch.round(outputs)  # Round to 0 or 1

            # Accuracy calculation
            total += targets.size(0)
            correct += (predicted_classes == targets).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy * 100:.2f}%')
        wandb.log({"test_accuracy": accuracy})
    torch.save(model.state_dict(), working_directory+'parameters/'+str(epoch)+".torch")
    #print("Save model to:",working_directory+'parameters/'+str(epoch)+".torch")