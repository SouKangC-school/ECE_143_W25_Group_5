import matplotlib.pyplot as plt
import numpy as np
def meanHistograms(ax, data, data_name):
    ax.hist(data, bins=25, color='skyblue', alpha=0.7, label='Data Distribution')

    data_mean = data.mean()
    data_std = data.std()

    # Plot vertical lines for mean and std deviations
    ax.axvline(data_mean, color='r', linestyle='--', label=f'Mean: {data_mean:.2f}')
    ax.axvline(data_mean + data_std, color='g', linestyle='--', label=f'+1 Std Dev: {data_mean + data_std:.2f}')
    ax.axvline(data_mean - data_std, color='b', linestyle='--', label=f'-1 Std Dev: {data_mean - data_std:.2f}')
    ax.set_xlabel("Heart Rate (BPM)", fontsize='small')
    ax.set_ylabel("Frequency", fontsize='small')
    ax.set_title(data_name)
    ax.legend(fontsize=8)

    return data_mean, data_std

