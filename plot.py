import matplotlib.pylab as plt
import numpy as np
import pandas as pd

def do_av(path:str, len:int, points:int):
    stack = np.zeros((1,points))
    for i in range(len):
        _path = path + str(i) + ".csv"
        data = np.genfromtxt(_path,delimiter=',')
        stack = np.vstack((stack, data))
    stack = stack[1::]
    return np.mean(stack,axis=0)


LEN = 300
POINTS = 247

plt.plot(do_av(".\\data2\\clean_Al_",LEN,POINTS),color="black")
plt.plot(do_av(".\\data2\\laser_blue_Al_",LEN,POINTS),color="blue")
plt.plot(do_av(".\\data2\\laser_Al_",LEN,POINTS),color="green")

plt.show()

