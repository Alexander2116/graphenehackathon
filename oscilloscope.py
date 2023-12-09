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

inst = pymeasure.instruments.keysight.KeysightDSOX1102G('USB0::0x2A8D::0x1797::CN57246135::INSTR')

inst.run()
inst.acquisition_mode = "realtime"
csv_path = ".\\data\\"
for i in range(50):
    csv_name ="clean" + str(i) + ".csv"
    data: np.array = inst.download_data("channel1",250)[0]
    np.savetxt(csv_path+csv_name,data)
inst.stop()