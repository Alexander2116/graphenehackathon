import matplotlib.pylab as plt
import numpy as np
import pandas as pd

for i in range(100):
    path = ".\\data\\clean_Al_" + str(i) + ".csv"
    data = pd.read_csv(path)
    plt.plot(data,color="blue")

for i in range(100):
    path = ".\\data\\laser_Al_" + str(i) + ".csv"
    data = pd.read_csv(path)
    plt.plot(data,color="green")

plt.show()