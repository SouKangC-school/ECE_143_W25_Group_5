"""
Preprocessing utilities for workout data.
"""
import numpy as np
from tqdm import tqdm

def extract_workout_data(workout):
    """
    Extract key metrics from a workout and return as a dictionary.
    
    Args:
        workout: Dictionary containing workout data
        
    Returns:
        Dictionary of extracted workout metrics
    """
    # Basic metadata
    workout_data = {
        'user_id': workout.get('userId', 'unknown'),
        'sport': workout.get('sport', 'unknown'),
        'gender': workout.get('gender', 'unknown'),
        'workout_id': workout.get('id', 'unknown')
    }
    
    # Extract heart rate data
    if 'tar_heart_rate' in workout and len(workout['tar_heart_rate']) > 0:
        hr_values = workout['tar_heart_rate']
        # Filter out likely invalid heart rate values (e.g., zeros or extremely high values)
        valid_hr = [hr for hr in hr_values if 30 < hr < 220]
        
        if valid_hr:
            workout_data.update({
                'avg_hr': np.mean(valid_hr),
                'max_hr': np.max(valid_hr),
                'min_hr': np.min(valid_hr),
                'hr_std': np.std(valid_hr),
                'hr_values': valid_hr,
                'hr_count': len(valid_hr)
            })
    
    # Extract speed data
    if 'tar_derived_speed' in workout and len(workout['tar_derived_speed']) > 0:
        speed_values = workout['tar_derived_speed']
        # Filter out negative or unreasonably high speeds
        valid_speed = [s for s in speed_values if 0 <= s < 30]  # 30 m/s = 108 km/h
        
        if valid_speed:
            workout_data.update({
                'avg_speed': np.mean(valid_speed),
                'max_speed': np.max(valid_speed),
                'speed_std': np.std(valid_speed),
                'speed_values': valid_speed,
                'speed_count': len(valid_speed)
            })
    
    # Extract distance data
    if 'distance' in workout:
        # Calculate total distance
        distance_values = workout['distance']
        if len(distance_values) > 1:
            # Use the maximum value as the total distance
            total_distance = np.max(distance_values)
            workout_data['total_distance'] = total_distance
    
    # Extract timestamp/duration data
    if 'timestamp' in workout:
        timestamps = workout['timestamp']
        if len(timestamps) > 1:
            # Calculate duration in seconds
            duration = timestamps[-1] - timestamps[0]
            workout_data['duration'] = duration
    
    # Extract altitude data if available
    if 'altitude' in workout:
        altitude_values = workout['altitude']
        valid_alt = [a for a in altitude_values if not np.isnan(a)]
        
        if valid_alt:
            alt_diff = np.diff(valid_alt)
            total_ascent = sum(diff for diff in alt_diff if diff > 0)
            total_descent = sum(abs(diff) for diff in alt_diff if diff < 0)
            
            workout_data.update({
                'total_ascent': total_ascent,
                'total_descent': total_descent,
                'altitude_range': np.max(valid_alt) - np.min(valid_alt),
            })
    
    return workout_data

def process_workouts(data):
    """
    Process all workouts to extract key metrics.
    
    Args:
        data: List of workout dictionaries
        
    Returns:
        List of processed workout dictionaries with extracted metrics
    """
    print("Extracting workout data...")
    all_workouts = []
    skipped_count = 0
    
    # Process each workout
    for i, workout in enumerate(data):
        if i % 1000 == 0:
            print(f"Processing workout {i}/{len(data)}...", end='\r')
        
        # Extract workout data
        workout_data = extract_workout_data(workout)
        
        # Only add workouts with heart rate data
        if 'avg_hr' in workout_data:
            all_workouts.append(workout_data)
        else:
            skipped_count += 1
    
    print(f"\nProcessed {len(all_workouts)} workouts with valid heart rate data (skipped {skipped_count})")
    return all_workouts

def prepare_sequences_optimized(data, seq_length=10, pred_horizon=1, features=None,
                               target='heart_rate', sample_ratio=0.2, max_users=100,
                               max_sequences_per_user=100):
    """
    Prepare sequences for training time series models with sampling to reduce dataset size.

    Args:
        data: List of workout dictionaries
        seq_length: Length of input sequences
        pred_horizon: How many steps ahead to predict
        features: List of feature names to include
        target: Target variable to predict
        sample_ratio: Ratio of data to sample
        max_users: Maximum number of users to include
        max_sequences_per_user: Maximum sequences to extract per user

    Returns:
        X: Input sequences
        y: Target values
    """
    if features is None:
        features = ['heart_rate', 'derived_speed', 'distance']

    X_sequences = []
    y_targets = []

    # Sample users
    unique_users = list(set(w.get('userId', i) for i, w in enumerate(data)))
    if len(unique_users) > max_users:
        unique_users = np.random.choice(unique_users, max_users, replace=False)

    user_workouts = {}
    for user_id in unique_users:
        user_workouts[user_id] = []

    # Group workouts by user
    for workout in data:
        user_id = workout.get('userId', None)
        if user_id in user_workouts:
            user_workouts[user_id].append(workout)

    # Process workouts for each sampled user
    for user_id, workouts in tqdm(user_workouts.items(), desc="Preparing sequences"):
        # Sample workouts for this user
        if len(workouts) > 10:  # Only sample if user has enough workouts
            workouts = np.random.choice(workouts, min(len(workouts), 10), replace=False)

        sequences_per_user = 0

        for workout in workouts:
            # Skip if target or any feature is missing
            if not all(f in workout for f in features + [target]):
                continue

            # Make sure all required features have same length
            feature_lens = [len(workout[f]) for f in features + [target]]
            if len(set(feature_lens)) > 1:  # Not all same length
                continue

            seq_len = feature_lens[0]

            # Skip if sequence is too short
            if seq_len <= seq_length + pred_horizon:
                continue

            # Sample sequence start points instead of using all
            # For longer sequences, take fewer samples to avoid bias
            stride = max(1, int(1 / sample_ratio))
            start_points = list(range(0, seq_len - seq_length - pred_horizon + 1, stride))

            # Limit sequences per workout
            if len(start_points) > max_sequences_per_user // len(workouts):
                start_points = np.random.choice(
                    start_points,
                    size=max_sequences_per_user // len(workouts),
                    replace=False
                )

            # Extract sequences at selected points
            for i in start_points:
                # Create feature vector for this sequence
                x_seq = []
                for f in features:
                    x_seq.append(workout[f][i:i+seq_length])

                # Stack features
                x_seq = np.column_stack(x_seq)
                X_sequences.append(x_seq)

                # Get target value
                y_targets.append(workout[target][i+seq_length+pred_horizon-1])

                sequences_per_user += 1
                if sequences_per_user >= max_sequences_per_user:
                    break

            if sequences_per_user >= max_sequences_per_user:
                break

    return np.array(X_sequences), np.array(y_targets)
