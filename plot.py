import matplotlib.pylab as plt
import numpy as np
import pandas as pd

for i in range(100):
    path = ".\\data2\\clean_Al_" + str(i) + ".csv"
    data = pd.read_csv(path)
    plt.plot(data,color="black")

"""for i in range(100):
    path = ".\\data2\\laser_blue_Al_" + str(i) + ".csv"
    data = pd.read_csv(path)
    plt.plot(data,color="blue")
"""
for i in range(100):
    path = ".\\data2\\laser_Al_" + str(i) + ".csv"
    data = pd.read_csv(path)
    plt.plot(data,color="green")

plt.show()