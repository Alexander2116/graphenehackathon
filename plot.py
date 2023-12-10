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
"""
plt.plot(do_av(".\\data2\\clean_Al_",LEN,POINTS),marker='o', linestyle='dashed',color="black",label = "no laser")
plt.plot(do_av(".\\data2\\laser_blue_Al_",LEN,POINTS),marker='o', linestyle='dashed',color="blue", label = "blue")
plt.plot(do_av(".\\data2\\laser_Al_",LEN,POINTS),marker='o', linestyle='dashed',color="green", label = "green")
plt.title("graphene_Al")
plt.ylabel("signal [V]")"""
plt.plot(do_av(".\\data3\\clean_Al_enc_print_",LEN,POINTS),marker='o', linestyle='dashed',color="black",label = "no laser")
plt.plot(do_av(".\\data3\\laser_blue_Al_enc_print_",LEN,POINTS),marker='o', linestyle='dashed',color="blue", label = "blue")
plt.plot(do_av(".\\data3\\laser_green_Al_enc_print_",LEN,POINTS),marker='o', linestyle='dashed',color="green", label = "green")
plt.title("graphene")
plt.ylabel("signal [V]")


plt.legend()
plt.show()

