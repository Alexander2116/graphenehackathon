import glob
import numpy as np
import matplotlib.pyplot
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

no_laser_files = glob.glob('Documents/graphene_hackathon/real_data/train/clean_Al*.csv')
no_laser_data = np.zeros((300,247))

for i,file in enumerate(no_laser_files):
	data = np.genfromtxt(file,delimiter=',')
	no_laser_data[i,::] = data

no_laser_avg_peak_2_peak = np.zeros((300))

for i in range(300):
	peaks, _ = find_peaks(no_laser_data[i,::], height=0)
	inv_peaks, _ = find_peaks(-no_laser_data[i,::], height=0)
	inv_values = -no_laser_data[0,::][inv_peaks]
	values = no_laser_data[0,::][peaks]
	peak_2_peak = values + inv_values
	no_laser_avg_peak_2_peak[i] = np.mean(peak_2_peak)

green_laser_files = glob.glob('Documents/graphene_hackathon/real_data/train/laser_Al*.csv')
green_laser_data = np.zeros((300,247))

for i,file in enumerate(green_laser_files):
	data = np.genfromtxt(file,delimiter=',')
	green_laser_data[i,::] = data

green_laser_avg_peak_2_peak = np.zeros((300))

for i in range(300):
	peaks, _ = find_peaks(green_laser_data[i,::], height=0)
	inv_peaks, _ = find_peaks(-green_laser_data[i,::], height=0)
	inv_values = -green_laser_data[0,::][inv_peaks]
	values = green_laser_data[0,::][peaks]
	peak_2_peak = values + inv_values
	green_laser_avg_peak_2_peak[i] = np.mean(peak_2_peak)

blue_laser_files = glob.glob('Documents/graphene_hackathon/real_data/train/laser_blue_Al*.csv')
blue_laser_data = np.zeros((300,247))

for i,file in enumerate(blue_laser_files):
	data = np.genfromtxt(file,delimiter=',')
	blue_laser_data[i,::] = data

blue_laser_avg_peak_2_peak = np.zeros((300))

for i in range(300):
	peaks, _ = find_peaks(blue_laser_data[i,::], height=0)
	inv_peaks, _ = find_peaks(-blue_laser_data[i,::], height=0)
	inv_values = -blue_laser_data[0,::][inv_peaks]
	values = blue_laser_data[0,::][peaks]
	peak_2_peak = values + inv_values
	blue_laser_avg_peak_2_peak[i] = np.mean(peak_2_peak)

cleaned_no_laser_peak_2_peak = no_laser_avg_peak_2_peak[no_laser_avg_peak_2_peak>np.mean(no_laser_avg_peak_2_peak)]
cleaned_green_laser_peak_2_peak = green_laser_avg_peak_2_peak[green_laser_avg_peak_2_peak>np.mean(green_laser_avg_peak_2_peak)]
cleaned_blue_laser_peak_2_peak = blue_laser_avg_peak_2_peak[blue_laser_avg_peak_2_peak>np.mean(blue_laser_avg_peak_2_peak)]
cleaned_cleaned_no_laser_peak_2_peak = cleaned_no_laser_peak_2_peak[cleaned_no_laser_peak_2_peak>np.mean(cleaned_no_laser_peak_2_peak)]


plt.figure(figsize=(12,8))
plt.plot(np.arange(0,len(cleaned_cleaned_no_laser_peak_2_peak[:140])),cleaned_cleaned_no_laser_peak_2_peak[:140],'ko',label='Without laser')
plt.plot(np.arange(0,len(cleaned_green_laser_peak_2_peak[:140])),cleaned_green_laser_peak_2_peak[:140],'go',label='Green laser')
plt.plot(np.arange(0,len(cleaned_blue_laser_peak_2_peak[:140])),cleaned_blue_laser_peak_2_peak[:140],'bo',label='Blue laser')
plt.xlabel('Number of runs')
plt.ylabel("Average peak to peak (V)")
plt.legend()
plt.title('Graphene/Aluminium/Graphene')
plt.show()



