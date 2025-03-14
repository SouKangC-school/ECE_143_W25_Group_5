import numpy as np

male_run_data = []
male_bike_data = []
female_run_data = []
female_bike_data = []
key_list = ['speed', 'gender', 'sport', 'userId', 'heart_rate', 'timestamp', 'id', 'altitude']

data = np.load('processed_endomondoHR_proper_interpolate.npy', allow_pickle=True)[0]
male_run_list=[]
female_run_list = []
for i in range (len(data)):
    if data[i]["sport"]=="run" and data[i]["gender"]=="male":
        del data[i]["longitude"]
        del data[i]["latitude"]
        del data[i]["since_last"]
        del data[i]["since_begin"]
        male_run_list.append(data[i])
    elif data[i]["sport"]=="run" and data[i]["gender"]=="female":
        del data[i]["longitude"]
        del data[i]["latitude"]
        del data[i]["since_last"]
        del data[i]["since_begin"]
        female_run_list.append(data[i])

male_run_list=np.array(male_run_list)
female_run_list = np.array(female_run_list)
np.save("male_run_data_with_proper_interpolate.npy",male_run_list)
np.save("female_run_data_with_proper_interpolate.npy",female_run_list)
