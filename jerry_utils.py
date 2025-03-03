import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import os
from fitparse import FitFile
from collections import defaultdict
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans



def draw_speed_distribution_with_fit(data, min_speed=0, max_speed=90,type="Run"):
    avg_speed = []
    for record in data:
        length = len(record["speed"])
        filtered_speed = record["speed"][int(length * 0.1):int(length * 0.9)]
        filtered_speed = [x for x in filtered_speed if min_speed < x < max_speed]
        mean_speed = np.mean(filtered_speed)
        if not np.isfinite(mean_speed) or np.isnan(mean_speed):
            continue
        avg_speed.append(mean_speed)

    mu, std = norm.fit(avg_speed)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    counts, bins, _ = ax1.hist(avg_speed, bins=50, edgecolor='black', linewidth=1.2, alpha=0.6)
    ax1.set_xlabel("Average Speed")
    ax1.set_ylabel("Frequency", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax2.plot(x, p, 'r', linewidth=2, label=f'Normal Fit\nMean = {mu:.2f}, Std = {std:.2f}')
    ax2.set_ylabel("Density", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title(type+" Average Speed Distribution with Normal Fit")
    ax2.legend(loc='upper right')

    plt.show()

    return avg_speed, mu, std


def remove_abnormal_speed_heartrate(_df):
    _df = _df[(_df['speed'] <= 20) & (_df['speed'] > 5) & (_df['heart_rate'] > 30)]
    return _df


def prediction_and_plot(_df,n=50000):
    _df = remove_abnormal_speed_heartrate(_df)
    X = _df['speed']  
    y = _df['heart_rate']  

    sampled_indices = _df.sample(n=n, random_state=42).index

    X = X[sampled_indices]
    y = y[sampled_indices]
    X= X.values.reshape(-1, 1)
    y = y.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)

    print(f"slope: {model.coef_[0]}")
    print(f"intercept: {model.intercept_}")

    # 进行预测
    y_pred = model.predict(X)

    plt.scatter(X, y, color='blue', label='Actual Heart Rate')
    plt.plot(X, y_pred, color='red', label='Linear regression line')
    plt.xlabel('Speed')
    plt.ylabel('Heart Rate')
    plt.text(0.05, 0.95, f'Slope: {model.coef_[0][0]:.2f}\nIntercept: {model.intercept_[0]:.2f}',
         transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.legend()
    plt.title('Speed vs Heart Rate - Linear Regression')
    plt.show()


def read_all_fit_files(folder_path):
    record_dict = defaultdict(dict)
    for filename in os.listdir(folder_path):
        if filename.endswith('.fit'):
            file_path = os.path.join(folder_path, filename)
            try:
                fitfile = FitFile(file_path, check_crc=False)
                heart_rate_list = []
                speed_list = []
                distance_list = []
                timestamp_list = []
                date_str = None
                for record in fitfile.get_messages('record'):
                    record_data = {d.name: d.value for d in record}
                    if record_data.get('activity_type') == 'running' and 'heart_rate' in record_data and 'speed' in record_data and 'timestamp' in record_data and 'distance' in record_data:
                        try:
                            timestamp = record_data['timestamp']
                            heart_rate = record_data['heart_rate']
                            speed = record_data['speed']
                            distance = record_data['distance']
                            if heart_rate is None or speed is None or distance is None:
                                continue
                            # Store daily data
                            timestamp_list.append(timestamp)
                            heart_rate_list.append(heart_rate)
                            speed_list.append(speed)
                            distance_list.append(distance)
                            date_str = timestamp.date().isoformat()
                        except Exception as e:
                            pass
                record_dict[date_str] = {"heart_rate": heart_rate_list, "speed": speed_list, "distance": distance_list, "timestamp": timestamp_list, "date": date_str,distance:distance_list[-1]}
            except Exception as e:
                pass
    return record_dict


def draw_random_samples_plot(_df, n_samples=100):
    plt.figure(figsize=(10, 6))
    df_male = _df[_df['gender'] == 'male']
    df_female = _df[_df['gender'] == 'female']
    
    df_male_sampled = df_male.sample(n=1, replace=True, )
    df_female_sampled = df_female.sample(n=1, replace=True)
    df_sampled = pd.concat([df_male_sampled, df_female_sampled])
    for index, row in df_sampled.iterrows():
        timestamps = row['timestamp']
        heart_rate = row['heart_rate']
        speed = row['speed']
        hr_color= "blue" if row["gender"] =="male" else "green"
        speed_color= "orange" if row["gender"] =="male" else "orange"
        label= "male" if row["gender"] =="male" else "female"
        time_diff = [ts - timestamps[0] for ts in timestamps]
        
        plt.plot(time_diff, heart_rate , marker='o',color=hr_color,label=label+" heart_rate")
        
        plt.plot(time_diff, speed, marker='x',color=speed_color,label=label+" speed")

    plt.title("Speed Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Heart Rate / Speed")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
from sklearn.linear_model import LinearRegression

def remove_abnormal_speed_heartrate(_df):
    _df = _df[(_df['speed'] <= 20) & (_df['speed'] > 5) & (_df['heart_rate'] > 30)]
    return _df


def prediction_and_plot(_df,n=50000):
    _df = remove_abnormal_speed_heartrate(_df)
    X = _df['speed']  
    y = _df['heart_rate']  

    sampled_indices = _df.sample(n=n, random_state=42).index

    X = X[sampled_indices]
    y = y[sampled_indices]
    X= X.values.reshape(-1, 1)
    y = y.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)

    print(f"slope: {model.coef_[0]}")
    print(f"intercept: {model.intercept_}")

    y_pred = model.predict(X)

    plt.scatter(X, y, color='blue', label='Actual Heart Rate')
    plt.plot(X, y_pred, color='red', label='Linear regression line')
    plt.xlabel('Speed')
    plt.ylabel('Heart Rate')
    plt.legend()
    plt.title('Speed vs Heart Rate - Linear Regression')
    plt.show()

def basemodel_plot():
    run_data = np.load("raw_run_data.npy",allow_pickle=True)
    run_df = pd.DataFrame(list(run_data))
    def pre_handle(df):
        """
        data clean
        calculate average speed and average heart rate for each record
        """
        df = df[df['gender'].isin(['male', 'female'])]
        df["avg_speed"] = df["speed"].apply(np.mean)
        df["avg_heart_rate"] = df["heart_rate"].apply(np.mean)
        # df["total_time"] = df["timestamp"].apply(lambda l: l[-1] - l[0])
        # df["distance"] = df.apply(lambda row: calculate_distance(row["timestamp"], row["speed"]), axis=1)
        summary_stats = df.groupby("gender")[["avg_heart_rate", "avg_speed"]].mean()
        print(summary_stats)
        return df

    run_df = pre_handle(run_df)
    draw_random_samples_plot(run_df)
    male_data = run_df[run_df["gender"] =="male"]
    female_data=run_df[run_df["gender"] =="female"]
    male_flat_df=pd.DataFrame({
        'speed': male_data['speed'].explode().reset_index(drop=True),
        'heart_rate': male_data['heart_rate'].explode().reset_index(drop=True),
    })
    female_flat_df=pd.DataFrame({
        'speed': female_data['speed'].explode().reset_index(drop=True),
        'heart_rate': female_data['heart_rate'].explode().reset_index(drop=True),
    })

    prediction_and_plot(male_flat_df,300)
    prediction_and_plot(female_flat_df,300)

# classification plot
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def classify_users_by_performance(data_list,speed_threshold=12 ,n_clusters=3):
    user_features = []

    for user_data in data_list:
        userId = user_data['userId']
        timestamps = np.array(user_data['timestamp'])
        heart_rate = np.array(user_data['heart_rate'])
        speed = np.array(user_data['derived_speed'])
        mean_speed = np.mean(user_data['tar_derived_speed'])
        user_data['mean_speed'] = mean_speed
        if mean_speed < speed_threshold:
            continue
        # time_diff = np.diff(timestamps, prepend=timestamps[0] + 1)
        speed_diff = np.diff(speed, prepend=speed[0])
        hr_diff = np.diff(heart_rate, prepend=heart_rate[0])

        valid_indices =[]
        for i in range(len(speed_diff)):
            if speed_diff[i] * hr_diff[i] > 0 and np.abs(speed_diff[i]) > 0.01:
                valid_indices.append(i)

        hr_speed_ratio = hr_diff[valid_indices] / speed_diff[valid_indices]

        hr_speed_ratio = hr_speed_ratio[~np.isnan(hr_speed_ratio)]
        hr_speed_ratio = hr_speed_ratio[np.isfinite(hr_speed_ratio)]
        
        if len(hr_speed_ratio) > 0:
            user_features.append({'userId': userId, 'mean_ratio': np.mean(hr_speed_ratio)})


    df_features = pd.DataFrame(user_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_features['cluster'] = kmeans.fit_predict(df_features[['mean_ratio']])

    cluster_mean = df_features.groupby('cluster')['mean_ratio'].mean().sort_values()
    mapping = {cluster: rank for rank, cluster in enumerate(cluster_mean.index)}
    df_features['performance_level'] = df_features['cluster'].map(mapping)
    level_map = {0: 'High', 1: 'Moderate', 2: 'Low'}
    df_features['performance_level'] = df_features['performance_level'].map(level_map)

    classified_users = {'High': [], 'Moderate': [], 'Low': []}
    for _, row in df_features.iterrows():
        classified_users[row['performance_level']].append(row)

    plt.figure(figsize=(8, 5))
    colors = {'High': 'green', 'Moderate': 'orange', 'Low': 'red'}
    for level, group in df_features.groupby('performance_level'):
        plt.plot(group['mean_ratio'], [0] * len(group), color=colors[level], label=level, alpha=0.7)

    plt.xlabel('Mean HR/Speed Ratio')
    plt.yticks([])
    plt.title('Performance Level Clustering')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

    return classified_users, df_features


def performance_classification_plot():
    data = np.load("run_data_with_proper_interpolate.npy",allow_pickle=True)
    classified_users, user_details = classify_users_by_performance(data, speed_threshold=11.6)
    high_performance_users = np.array(classified_users['High'])
    hp_id_map = {user[0]: user for user in high_performance_users}
    hp_data=[]
    for user in data:
        if user['userId'] in hp_id_map and 14 > user['mean_speed'] > 10:
            hp_data.append({"speed":user["tar_derived_speed"],'heart_rate':user["tar_heart_rate"]})
    hp_df=pd.DataFrame(hp_data)
    hp_df_flat=pd.DataFrame({
        'speed': hp_df['speed'].explode().reset_index(drop=True),
        'heart_rate': hp_df['heart_rate'].explode().reset_index(drop=True),
    })
    prediction_and_plot(hp_df_flat,300 )
    draw_speed_distribution_with_fit(hp_data,min_speed=6,max_speed=24, type="High Performance Users")
    moderate_performance_users = np.array(classified_users['Moderate'])
    mp_id_map = {user[0]: user for user in moderate_performance_users}
    mp_data=[]
    for user in data:
        if user['userId'] in mp_id_map and  14> user['mean_speed'] > 10:
            mp_data.append({"speed":user["tar_derived_speed"],'heart_rate':user["tar_heart_rate"]})
    mp_df=pd.DataFrame(mp_data)
    mp_df_flat=pd.DataFrame({
        'speed':mp_df['speed'].explode().reset_index(drop=True),
        'heart_rate': mp_df['heart_rate'].explode().reset_index(drop=True),
    })
    prediction_and_plot(mp_df_flat,300 )
    draw_speed_distribution_with_fit(mp_data,min_speed=6,max_speed=24, type="Moderate Performance Users")
    low_performance_users = np.array(classified_users['Low'])
    lp_id_map = {user[0]: user for user in low_performance_users}
    lp_data=[]
    for user in data:
        if user['userId'] in lp_id_map and 14> user['mean_speed'] > 10:
            lp_data.append({"speed":user["tar_derived_speed"],'heart_rate':user["tar_heart_rate"]})
    lp_df=pd.DataFrame(lp_data)
    lp_df_flat=pd.DataFrame({
        'speed': lp_df['speed'].explode().reset_index(drop=True),
        'heart_rate': lp_df['heart_rate'].explode().reset_index(drop=True),
    })
    prediction_and_plot(lp_df_flat,300 )
    draw_speed_distribution_with_fit(lp_data,min_speed=6,max_speed=24, type="Low Performance Users")

    male_data=[]
    female_data=[]
    for user in data:
        if user['gender'] == 'male':
            male_data.append(user)
        elif user['gender'] == 'female':
            female_data.append(user)
    classified_male_users, male_user_details = classify_users_by_performance(male_data, speed_threshold=11.6,n_clusters=3)
    for level in ['High', 'Moderate','Low']:
        print(f"{level} performance")
        performance_users = np.array(classified_male_users[level])
        id_map = {user[0]: user for user in performance_users}
        p_data=[]
        for user in male_data:
            if user['userId'] in id_map and  14> user['mean_speed'] > 10:
                p_data.append({"speed":user["tar_derived_speed"],'heart_rate':user["tar_heart_rate"]})
        p_df=pd.DataFrame(p_data)
        p_df_flat=pd.DataFrame({
            'speed': p_df['speed'].explode().reset_index(drop=True),
            'heart_rate': p_df['heart_rate'].explode().reset_index(drop=True),
        })
        prediction_and_plot(p_df_flat,300 )
        draw_speed_distribution_with_fit(p_data,min_speed=6,max_speed=24, type=f"{level} Performance Users")
    
    classified_female_users, female_user_details = classify_users_by_performance(female_data,speed_threshold=9.92,n_clusters=3)
    for level in ['High', 'Moderate','Low']:
        print(f"{level} performance")
        performance_users = np.array(classified_female_users[level])
        id_map = {user[0]: user for user in performance_users}
        p_data=[]
        for user in female_data:
            if user['userId'] in id_map and 12> user['mean_speed'] > 9 and user["tar_derived_speed"] is not None:
                p_data.append({"speed":user["tar_derived_speed"],'heart_rate':user["tar_heart_rate"]})
        p_df=pd.DataFrame(p_data)
        p_df=p_df.dropna()
        p_df_flat=pd.DataFrame({
            'speed': p_df['speed'].explode().reset_index(drop=True),
            'heart_rate': p_df['heart_rate'].explode().reset_index(drop=True),
        })
        prediction_and_plot(p_df_flat,300 )
        draw_speed_distribution_with_fit(p_data,min_speed=6,max_speed=24, type=f"{level} Performance Users")
    
    for level in ['High', 'Moderate','Low']:
        print(f"{level} performance")
        performance_users = np.array(classified_male_users[level])
        id_map = {user[0]: user for user in performance_users}
        p_data=[]
        for user in male_data:
            if user['userId'] in id_map and  16> user['mean_speed'] > 14:
                p_data.append({"speed":user["tar_derived_speed"],'heart_rate':user["tar_heart_rate"]})
        p_df=pd.DataFrame(p_data)
        p_df_flat=pd.DataFrame({
            'speed': p_df['speed'].explode().reset_index(drop=True),
            'heart_rate': p_df['heart_rate'].explode().reset_index(drop=True),
        })
        prediction_and_plot(p_df_flat,300 )
        draw_speed_distribution_with_fit(p_data,min_speed=6,max_speed=24, type=f"{level} Performance Users")

    classified_female_users, female_user_details = classify_users_by_performance(female_data,speed_threshold=9.92,n_clusters=3)

    for level in ['High', 'Moderate','Low']:
        print(f"{level} performance")
        performance_users = np.array(classified_female_users[level])
        id_map = {user[0]: user for user in performance_users}
        p_data=[]
        for user in female_data:
            if user['userId'] in id_map and 14> user['mean_speed'] > 12 and user["tar_derived_speed"] is not None:
                p_data.append({"speed":user["tar_derived_speed"],'heart_rate':user["tar_heart_rate"]})
        p_df=pd.DataFrame(p_data)
        p_df=p_df.dropna()
        p_df_flat=pd.DataFrame({
            'speed': p_df['speed'].explode().reset_index(drop=True),
            'heart_rate': p_df['heart_rate'].explode().reset_index(drop=True),
        })
        prediction_and_plot(p_df_flat,300 )
        draw_speed_distribution_with_fit(p_data,min_speed=6,max_speed=24, type=f"{level} Performance Users")

def draw_performance_indicator_comparison(M_male,M_female,label_male,label_female,speed_level):
    categories = ["High", "Moderate", "Low"]
    x = np.arange(len(categories))
    bar_width = 0.3
    plt.figure(figsize=(8, 5))
    plt.bar(x - bar_width / 2, M_male, width=bar_width, color='#0072B2', label=label_male)
    plt.bar(x + bar_width / 2, M_female, width=bar_width, color='#D55E00', label=label_female)
    for i in range(3):
        plt.text(x[i] - bar_width / 2, M_male[i] + 1, f"{M_male[i]:.2f}", ha='center', fontsize=10)
        plt.text(x[i] + bar_width / 2, M_female[i] + 1, f"{M_female[i]:.2f}", ha='center', fontsize=10)
    plt.xticks(x, categories)
    plt.ylabel("Performance Indicator (M=HR*exp(-S))")
    plt.title(f"{speed_level} Speed Performance Indicator Comparison in Gender")
    plt.legend()
    plt.ylim(0, max(max(M_male), max(M_female)) + 10)  
    plt.show()

def runner_indicator_plot():
    slope_male_10_14 = np.array([1.86, 2.2, 2.32])
    slope_male_14_16 = np.array([0.81, 1.23, 1.90])
    slope_female_9_12 = np.array([2.08, 2.51, 2.52])
    slope_female_12_14 = np.array([1.55, 1.13, 1.50])

    itercept_hr_male_10_14=np.array([124.03,121,115])
    itercept_hr_male_14_16=np.array([125.95,121.16,118.9])
    itercept_hr_female_9_12=np.array([147.34,150.94,130.57])
    itercept_hr_female_12_14=np.array([147.75,153.88,148])

    avg_hr_male_10_14=itercept_hr_male_10_14+12*slope_male_10_14
    avg_hr_male_14_16=itercept_hr_male_14_16+15*slope_male_14_16
    avg_hr_female_9_12=itercept_hr_female_9_12+10.5*slope_female_9_12
    avg_hr_female_12_14=itercept_hr_female_12_14+13*slope_female_12_14

    M_male_10_14=avg_hr_male_10_14*np.exp(-slope_male_10_14)
    M_male_14_16=avg_hr_male_14_16*np.exp(-slope_male_14_16)
    M_female_9_12=avg_hr_female_9_12*np.exp(-slope_female_9_12)
    M_female_12_14=avg_hr_female_12_14*np.exp(-slope_female_12_14)
    draw_performance_indicator_comparison(M_male_10_14,M_female_9_12,"Male 10-14km/h","Female 9-12km/h","Normal")
    draw_performance_indicator_comparison(M_male_14_16,M_female_12_14,"Male 14-16km/h","Female 12-14km/h","Competitive")