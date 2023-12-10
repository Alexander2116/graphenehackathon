import pyvisa
import pymeasure.instruments
import matplotlib.pyplot as plt
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob
import random
import torch.nn.functional as F
#from pymeasure import keysightDSOX1102G
# Should be GPIB
#keithley = rm.open_resource() 
# Should be GPIB
#keithley = rm.open_resource() 

def show_resources():
    rm = pyvisa.ResourceManager()
    print(rm.list_resources())
    
show_resources()


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


class Sensor():
    def __init__(self, visa_str:str = "USB0::0x2A8D::0x1797::CN57246135::INSTR"):
        self.instr = pymeasure.instruments.keysight.KeysightDSOX1102G(visa_str)
        self.instr.acquisition_mode = "realtime"
        self.instr.run()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        input_size = 247
        hidden_size = 128
        output_size = 3

        weigths_path = '.\\499.torch'
        self.model = LaserRegression(input_size, hidden_size, output_size)
        self.model.load_state_dict(torch.load(weigths_path))
        self.model.to(self.device)
        self.model.eval()
    
    def take_data(self, ch: str = "channel1", data_points: int = 250) -> np.array:
        #self.instr.run()
        data: np.array = self.instr.download_data(ch, data_points)[0]
        #self.instr.stop()
        return data
    
    def save_data(path:str, data:np.array):
        np.savetxt(path,data)

    def predict_laser(self):
        # ML model that predicts if the laser is on or off
        # return [on/off, r, g, b]
        data = self.take_data()
        data_tensor = torch.tensor(data, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(data_tensor)
            probabilities = F.softmax(outputs).detach().cpu().numpy()
            predicted_classes = np.argmax(probabilities)
            print('Predicted class',predicted_classes)


    def off(self):
        self.instr.stop()

#inst = Sensor()


"""
inst = pymeasure.instruments.keysight.KeysightDSOX1102G('USB0::0x2A8D::0x1797::CN57246135::INSTR')
inst.run()
inst.acquisition_mode = "realtime"
csv_path = ".\\data2\\"
for i in range(300):
    csv_name ="laser_blue_Al_" + str(i) + ".csv"
    data: np.array = inst.download_data("channel1",250)[0]
    np.savetxt(csv_path+csv_name,data)
inst.stop()
"""

instr = Sensor()

csv_path = ".\\data4\\"
for i in range(300):
    csv_name ="laser_blue_Al_" + str(i) + ".csv"
    data: np.array = instr.take_data("channel1",250)
    np.savetxt(csv_path+csv_name,data)

instr.off()

#instr.predict_laser()