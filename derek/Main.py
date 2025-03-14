import numpy as np
import matplotlib.pyplot as plt
import Histograms

male_run_data = np.load('male_run_data_with_proper_interpolate.npy', allow_pickle=True)
female_run_data = np.load('female_run_data_with_proper_interpolate.npy', allow_pickle=True)

female_speeds_to_hr = {'<10':[], '10-12':[], '12-14':[], '14-16':[], '>16': []}
male_speeds_to_hr = {'<10':[], '10-12':[], '12-14':[], '14-16':[], '>16': []}
for workout in male_run_data:

    speed_data = workout['tar_derived_speed'][int(len(workout['tar_derived_speed'])*.1) : int(len(workout['tar_derived_speed'])*.9)]
    heart_data = workout['tar_heart_rate'][int(len(workout['tar_heart_rate']) * .1): int(len(workout['tar_heart_rate']) * .9)]
    max_speed = max(speed_data)
    max_heart_rate = max(heart_data)
    if max_speed < 10:
        male_speeds_to_hr['<10'].append(max_heart_rate)
    elif max_speed < 12:
        male_speeds_to_hr['10-12'].append(max_heart_rate)
    elif max_speed < 14:
        male_speeds_to_hr['12-14'].append(max_heart_rate)
    elif max_speed < 16:
        male_speeds_to_hr['14-16'].append(max_heart_rate)
    else:
        male_speeds_to_hr['>16'].append(max_heart_rate)

for workout in female_run_data:

    speed_data = workout['tar_derived_speed'][int(len(workout['tar_derived_speed'])*.1) : int(len(workout['tar_derived_speed'])*.9)]
    heart_data = workout['tar_heart_rate'][int(len(workout['tar_heart_rate']) * .1): int(len(workout['tar_heart_rate']) * .9)]
    max_speed = max(speed_data)
    max_heart_rate = max(heart_data)
    if max_speed < 10:
        female_speeds_to_hr['<10'].append(max_heart_rate)
    elif max_speed < 12:
        female_speeds_to_hr['10-12'].append(max_heart_rate)
    elif max_speed < 14:
        female_speeds_to_hr['12-14'].append(max_heart_rate)
    elif max_speed < 16:
        female_speeds_to_hr['14-16'].append(max_heart_rate)
    else:
        female_speeds_to_hr['>16'].append(max_heart_rate)

def plot_multiple_histograms(speeds_to_hr, gender):
    num_plots = min(len(speeds_to_hr), 5)  # Limit to 5 subplots max
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns layout

    axes = axes.flatten()  # Flatten to 1D array for easy iteration

    for i, (speed, hr_data) in enumerate(speeds_to_hr.items()):
        if i >= 5:  # Only plot up to 5 histograms
            break
        hr_data = np.array(hr_data)  # Convert to NumPy array
        Histograms.meanHistograms(axes[i], hr_data, f"{speed} km/h")

    # Hide any unused subplots if there are fewer than 5 datasets
    for j in range(i + 1, 6):
        fig.delaxes(axes[j])
    plt.suptitle(f"{gender} speed to Heart Rate")
    plt.tight_layout()
    plt.show()

plot_multiple_histograms(female_speeds_to_hr, "Women")
plot_multiple_histograms(male_speeds_to_hr, "Men")

