import numpy as np
import matplotlib.pyplot as plt

Speed = np.array(['<10', '10-12', '12-14', '14-16', '>16'])
x_positions = np.arange(len(Speed))  # Convert categorical labels to numerical positions

heart_rate_men = np.array([[151.1, 17.6], [154.7, 14.4], [159.2, 14.0], [162.0, 13.8], [165.2, 13.6]])
heart_rate_women = np.array([[153.6, 14.7], [161.7, 13.2], [164.2, 13.7], [166.7, 13.9], [167.9, 13.8]])

# Plot error bars
plt.errorbar(x_positions, heart_rate_men[:, 0], yerr=heart_rate_men[:, 1], label='Men', fmt='o', capsize=5)
plt.errorbar(x_positions + 0.2, heart_rate_women[:, 0], yerr=heart_rate_women[:, 1], label='Women', fmt='o', capsize=5)

# Connect the means with lines
plt.plot(x_positions, heart_rate_men[:, 0], linestyle='-', marker='o', label="Men Line")
plt.plot(x_positions + 0.2, heart_rate_women[:, 0], linestyle='-', marker='o', label="Women Line")

# Set x-ticks back to categorical labels
plt.xticks(x_positions, Speed)
plt.xlabel('Speed (km/h)')
plt.ylabel('Heart Rate (bpm)')
plt.legend()
plt.show()