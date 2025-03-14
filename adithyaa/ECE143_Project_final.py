import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

run_data = []
bike_data = []
key_list = ['speed', 'gender', 'sport', 'userId', 'heart_rate', 'timestamp', 'id']

# Load and process the JSON data
with open('/content/drive/MyDrive/endomondoHR_proper.json') as f:
    for line in f:
        d = json.loads(line.replace("\'", "\""))
        if d.get('speed') is None or d.get('heart_rate') is None:
            continue
        if len(d['timestamp']) < 10:
            continue

        if d['sport'] == 'bike':
            bike_data.append({k: d[k] for k in key_list})
        elif d['sport'] == 'run':
            run_data.append({k: d[k] for k in key_list})

# Convert to numpy arrays and save
bike_data = np.array(bike_data)
run_data = np.array(run_data)
np.save("bike_data.npy", bike_data)
np.save("run_data.npy", run_data)

# Load the .npy files
bike_data = np.load("bike_data.npy", allow_pickle=True)
run_data = np.load("run_data.npy", allow_pickle=True)

# Convert to Pandas DataFrame
bike_df = pd.DataFrame(bike_data.tolist())
run_df = pd.DataFrame(run_data.tolist())

# Clean the heart_rate column
def flatten_heart_rate(value):
    if isinstance(value, list) and len(value) > 0:
        return value[0]  # Or you can take the average, e.g., return np.mean(value)
    return value

# Apply the flattening function to both bike and run data
bike_df['heart_rate'] = bike_df['heart_rate'].apply(flatten_heart_rate)
run_df['heart_rate'] = run_df['heart_rate'].apply(flatten_heart_rate)

# Convert heart_rate to numeric, coercing errors to NaN
bike_df['heart_rate'] = pd.to_numeric(bike_df['heart_rate'], errors='coerce')
run_df['heart_rate'] = pd.to_numeric(run_df['heart_rate'], errors='coerce')

# Check for remaining NaN values
print(bike_df['heart_rate'].isnull().sum(), "NaN values remaining in bike heart_rate")
print(run_df['heart_rate'].isnull().sum(), "NaN values remaining in run heart_rate")

# Calculate exercise frequency for bike and run data separately
bike_exercise_freq = bike_df['userId'].value_counts().reset_index()
run_exercise_freq = run_df['userId'].value_counts().reset_index()

bike_exercise_freq.columns = ['userId', 'exercise_freq']
run_exercise_freq.columns = ['userId', 'exercise_freq']

# Merge exercise frequency back to both bike and run DataFrames
bike_df = bike_df.merge(bike_exercise_freq, on='userId')
run_df = run_df.merge(run_exercise_freq, on='userId')

# Calculate average heart rate per user for bike and run data separately
bike_performance_metric = bike_df.groupby('userId')['heart_rate'].mean().reset_index()
run_performance_metric = run_df.groupby('userId')['heart_rate'].mean().reset_index()

bike_performance_metric.columns = ['userId', 'average_heart_rate']
run_performance_metric.columns = ['userId', 'average_heart_rate']

# Merge performance metric back to both DataFrames
bike_df = bike_df.merge(bike_performance_metric, on='userId')
run_df = run_df.merge(run_performance_metric, on='userId')

# Select relevant features for clustering (exercise_freq and average_heart_rate)
bike_features = bike_df[['exercise_freq', 'average_heart_rate']].drop_duplicates()
run_features = run_df[['exercise_freq', 'average_heart_rate']].drop_duplicates()

# Normalize the features
scaler = StandardScaler()
bike_features_scaled = scaler.fit_transform(bike_features)
run_features_scaled = scaler.fit_transform(run_features)

# Apply KMeans clustering to both datasets
bike_kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust n_clusters as needed
run_kmeans = KMeans(n_clusters=3, random_state=42)

bike_features['cluster'] = bike_kmeans.fit_predict(bike_features_scaled)
run_features['cluster'] = run_kmeans.fit_predict(run_features_scaled)

# Visualize the bike clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=bike_features, x='exercise_freq', y='average_heart_rate', hue='cluster', palette='viridis')
plt.title('Bike Exercise Frequency vs Performance (Average Heart Rate) Clusters')
plt.xlabel('Exercise Frequency')
plt.ylabel('Average Heart Rate')
plt.legend(title='Cluster')
plt.show()

# Visualize the run clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=run_features, x='exercise_freq', y='average_heart_rate', hue='cluster', palette='viridis')
plt.title('Run Exercise Frequency vs Performance (Average Heart Rate) Clusters')
plt.xlabel('Exercise Frequency')
plt.ylabel('Average Heart Rate')
plt.legend(title='Cluster')
plt.show()

# Optionally save the results
bike_df.to_csv('bike_data_with_clusters.csv', index=False)
run_df.to_csv('run_data_with_clusters.csv', index=False)

print("Analysis complete and results saved.")
