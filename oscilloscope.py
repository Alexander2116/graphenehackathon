import serial
import serial.tools.list_ports
import sys
import time
import pyvisa
#from pymeasure import keysightDSOX1102G
# Should be GPIB
#keithley = rm.open_resource() 
# Should be GPIB
#keithley = rm.open_resource() 

def show_resources():
    rm = pyvisa.ResourceManager()
    print(rm.list_resources())
    
show_resources()

