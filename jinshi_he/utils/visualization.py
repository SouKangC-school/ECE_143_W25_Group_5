"""
Visualization utilities for workout data analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def setup_visualization_style(style='ggplot', figsize=(12, 8), fontsize=12):
    """
    Set up the visualization style for plots.
    
    Args:
        style: Plot style
        figsize: Figure size as (width, height)
        fontsize: Font size for plot text
    """
    plt.style.use(style)
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = fontsize

def plot_workout_counts_by_sport(workouts_df):
    """
    Plot the count of workouts by sport type.
    
    Args:
        workouts_df: DataFrame containing workout data
    """
    sport_counts = workouts_df['sport'].value_counts()
    print("\nWorkouts by sport type:")
    print(sport_counts)
    
    plt.figure(figsize=(14, 8))
    sport_counts.plot(kind='bar', color='skyblue')
    plt.title('Number of Workouts by Sport Type', fontsize=16)
    plt.xlabel('Sport Type', fontsize=14)
    plt.ylabel('Number of Workouts', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_gender_distribution(workouts_df):
    """
    Plot the gender distribution of athletes.
    
    Args:
        workouts_df: DataFrame containing workout data
    """
    if 'gender' in workouts_df.columns:
        gender_counts = workouts_df['gender'].value_counts()
        plt.figure(figsize=(10, 6))
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
                colors=['#3498db', '#e74c3c', '#2ecc71'], explode=[0.05, 0.05, 0.05])
        plt.title('Gender Distribution of Athletes', fontsize=16)
        plt.tight_layout()
        plt.show()

def plot_heart_rate_by_sport(workouts_df, min_count=10):
    """
    Plot average heart rate by sport type.
    
    Args:
        workouts_df: DataFrame containing workout data
        min_count: Minimum number of workouts required to include sport
    """
    # Calculate average heart rate by sport
    hr_by_sport = workouts_df.groupby('sport')['avg_hr'].agg(['mean', 'std', 'count']).sort_values('count', ascending=False)
    hr_by_sport = hr_by_sport[hr_by_sport['count'] >= min_count]  # Filter out sports with too few samples

    # Create a color map for consistent sport colors
    sport_colors = {}
    color_palette = sns.color_palette('husl', n_colors=len(hr_by_sport))
    for i, sport in enumerate(hr_by_sport.index):
        sport_colors[sport] = color_palette[i]

    plt.figure(figsize=(14, 8))
    bars = plt.bar(hr_by_sport.index, hr_by_sport['mean'], 
                   yerr=hr_by_sport['std'], capsize=5, 
                   color=[sport_colors[s] for s in hr_by_sport.index],
                   alpha=0.7)

    plt.title('Average Heart Rate by Sport Type', fontsize=16)
    plt.xlabel('Sport Type', fontsize=14)
    plt.ylabel('Average Heart Rate (bpm)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return sport_colors

def classify_workout_intensity(avg_hr, max_hr=190):
    """
    Classify workout intensity based on heart rate.
    
    Args:
        avg_hr: Average heart rate
        max_hr: Maximum heart rate
        
    Returns:
        String representing intensity level
    """
    hr_percent = avg_hr / max_hr * 100
    if hr_percent < 60:
        return 'Very Light'
    elif hr_percent < 70:
        return 'Light'
    elif hr_percent < 80:
        return 'Moderate'
    elif hr_percent < 90:
        return 'Hard'
    else:
        return 'Very Hard'

def plot_intensity_distribution(workouts_df):
    """
    Plot the distribution of workout intensity levels.
    
    Args:
        workouts_df: DataFrame containing workout data
    """
    # Add intensity classification to the dataframe
    workouts_df['intensity'] = workouts_df['avg_hr'].apply(classify_workout_intensity)

    # Analyze intensity distribution
    intensity_order = ['Very Light', 'Light', 'Moderate', 'Hard', 'Very Hard']
    intensity_counts = workouts_df['intensity'].value_counts().reindex(intensity_order)

    plt.figure(figsize=(12, 7))
    bars = plt.bar(intensity_counts.index, intensity_counts.values, 
                   color=['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'])

    plt.title('Distribution of Workout Intensity', fontsize=16)
    plt.xlabel('Intensity Level', fontsize=14)
    plt.ylabel('Number of Workouts', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add percentage labels
    total = intensity_counts.sum()
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 5,
                 f'{height/total*100:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def plot_intensity_by_sport(workouts_df, sport_counts, min_count=50):
    """
    Plot the intensity distribution by sport type.
    
    Args:
        workouts_df: DataFrame containing workout data
        sport_counts: Series of sport counts
        min_count: Minimum count to include a sport
    """
    # Calculate the percentage of each intensity level for top sports
    top_sports = sport_counts[sport_counts >= min_count].index.tolist()  # Focus on sports with enough data
    intensity_order = ['Very Light', 'Light', 'Moderate', 'Hard', 'Very Hard']
    
    intensity_by_sport = pd.crosstab(
        workouts_df[workouts_df['sport'].isin(top_sports)]['sport'], 
        workouts_df[workouts_df['sport'].isin(top_sports)]['intensity'],
        normalize='index'
    ) * 100

    # Ensure all columns are present, even if some sports don't have certain intensities
    for level in intensity_order:
        if level not in intensity_by_sport.columns:
            intensity_by_sport[level] = 0

    # Reorder columns for consistent ordering
    intensity_by_sport = intensity_by_sport.reindex(columns=intensity_order)

    plt.figure(figsize=(14, 10))
    intensity_by_sport.plot(kind='bar', stacked=True, 
                          color=['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'])

    plt.title('Workout Intensity Distribution by Sport Type', fontsize=16)
    plt.xlabel('Sport Type', fontsize=14)
    plt.ylabel('Percentage of Workouts', fontsize=14)
    plt.legend(title='Intensity Level')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_hr_variability(workouts_df, top_sports, sport_colors):
    """
    Plot heart rate variability by sport type.
    
    Args:
        workouts_df: DataFrame containing workout data
        top_sports: List of top sports to include
        sport_colors: Dictionary mapping sports to colors
    """
    plt.figure(figsize=(12, 8))
    for sport in top_sports:
        sport_data = workouts_df[workouts_df['sport'] == sport]
        plt.scatter(sport_data['avg_hr'], sport_data['hr_std'], 
                    label=sport, color=sport_colors.get(sport, 'gray'), alpha=0.6)

    plt.title('Heart Rate Variability by Sport Type', fontsize=16)
    plt.xlabel('Average Heart Rate (bpm)', fontsize=14)
    plt.ylabel('Heart Rate Standard Deviation', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Sport Type')

    # Add trend line for the entire dataset
    z = np.polyfit(workouts_df['avg_hr'], workouts_df['hr_std'], 1)
    p = np.poly1d(z)
    plt.plot(sorted(workouts_df['avg_hr'].unique()), 
             p(sorted(workouts_df['avg_hr'].unique())), 
             'r--', linewidth=2, label='Trend Line')

    plt.tight_layout()
    plt.show()

def plot_speed_vs_hr(workouts_df, top_sports, sport_colors, focus_sports=None):
    """
    Plot average speed vs heart rate by sport type.
    
    Args:
        workouts_df: DataFrame containing workout data
        top_sports: List of top sports to include
        sport_colors: Dictionary mapping sports to colors
        focus_sports: List of sports to focus on (default: ['run', 'bike', 'walk'])
    """
    if focus_sports is None:
        focus_sports = ['run', 'bike', 'walk']
        
    # Filter for workouts that have both heart rate and speed data
    speed_hr_df = workouts_df.dropna(subset=['avg_hr', 'avg_speed'])

    plt.figure(figsize=(12, 8))
    for sport in top_sports:
        if sport in focus_sports:  # Focus on sports where speed is relevant
            sport_data = speed_hr_df[speed_hr_df['sport'] == sport]
            if len(sport_data) > 10:  # Only plot if we have enough data
                plt.scatter(sport_data['avg_hr'], sport_data['avg_speed'], 
                            label=f"{sport} (n={len(sport_data)})", 
                            color=sport_colors.get(sport, 'gray'), alpha=0.6)

    plt.title('Average Speed vs Heart Rate by Sport Type', fontsize=16)
    plt.xlabel('Average Heart Rate (bpm)', fontsize=14)
    plt.ylabel('Average Speed (m/s)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Sport Type')
    plt.tight_layout()
    plt.show()
    
    # Calculate correlation between HR and speed for each sport
    correlations = {}
    for sport in top_sports:
        if sport in focus_sports:
            sport_data = speed_hr_df[speed_hr_df['sport'] == sport]
            if len(sport_data) > 10:
                corr = sport_data['avg_hr'].corr(sport_data['avg_speed'])
                correlations[sport] = corr

    print("Correlation between heart rate and speed:")
    for sport, corr in correlations.items():
        print(f"{sport}: {corr:.3f}")
        
    return correlations

def plot_training_history(train_losses, val_losses=None, val_epochs=None, model_name="Model"):
    """
    Plot training and validation loss history.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        val_epochs: List of epochs for validation losses (optional)
        model_name: Name of the model for the plot title
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    
    if val_losses is not None and len(val_losses) > 0:
        if val_epochs is None:
            # If val_epochs not provided, assume validation was done every epoch
            plt.plot(val_losses, label='Validation Loss')
        else:
            # Ensure val_epochs and val_losses have the same length
            val_epochs = val_epochs[:len(val_losses)]
            plt.plot(val_epochs, val_losses, label='Validation Loss')
            
    plt.title(f'{model_name} Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(targets, predictions, model_name="Model", sample_size=500):
    """
    Plot predictions vs actual values.
    
    Args:
        targets: Array of actual target values
        predictions: Array of predicted values
        model_name: Name of the model for the plot title
        sample_size: Number of points to sample for the plot
    """
    plt.figure(figsize=(10, 6))

    # Take a sample of points for better visualization
    sample_size = min(sample_size, len(targets))
    indices = np.random.choice(len(targets), sample_size, replace=False)

    plt.scatter(
        targets[indices],
        predictions[indices],
        alpha=0.5,
        label='Predictions'
    )

    # Add perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    plt.title(f'{model_name} Predictions vs Actual')
    plt.xlabel('Actual Heart Rate')
    plt.ylabel('Predicted Heart Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
