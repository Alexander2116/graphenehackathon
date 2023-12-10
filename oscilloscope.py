import pyvisa
import pymeasure.instruments
import matplotlib.pyplot as plt
import time
import csv
import numpy as np
#from pymeasure import keysightDSOX1102G
# Should be GPIB
#keithley = rm.open_resource() 
# Should be GPIB
#keithley = rm.open_resource() 

def show_resources():
    rm = pyvisa.ResourceManager()
    print(rm.list_resources())
    
show_resources()


class Sensor():
    def __init__(self, visa_str:str = "USB0::0x2A8D::0x1797::CN57246135::INSTR"):
        self.instr = pymeasure.instruments.keysight.KeysightDSOX1102G('USB0::0x2A8D::0x1797::CN57246135::INSTR')
        self.instr.acquisition_mode = "realtime"
    
    def take_data(self, ch: str = "channel1", data_points: int = 250) -> np.array:
        self.instr.run()
        data: np.array = self.instr.download_data(ch, data_points)[0]
        self.instr.stop()
        return data
    
    def save_data(path:str, data:np.array):
        np.savetxt(path,data)

    def predict_laser(self) -> (int,int,int,int):
        # ML model that predicts if the laser is on or off
        # return [on/off, r, g, b]
        data = self.take_data()


#inst = Sensor()



inst = pymeasure.instruments.keysight.KeysightDSOX1102G('USB0::0x2A8D::0x1797::CN57246135::INSTR')
inst.run()
inst.acquisition_mode = "realtime"
csv_path = ".\\data2\\"
for i in range(300):
    csv_name ="laser_blue_Al_" + str(i) + ".csv"
    data: np.array = inst.download_data("channel1",250)[0]
    np.savetxt(csv_path+csv_name,data)
inst.stop()

