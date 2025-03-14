"""
Workout analysis functionality.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math
from scipy import stats
from collections import defaultdict
import warnings

from ..utils.data_loader import load_data, convert_to_dataframe
from ..utils.preprocessing import process_workouts
from ..utils.visualization import (
    setup_visualization_style,
    plot_workout_counts_by_sport,
    plot_gender_distribution,
    plot_heart_rate_by_sport,
    plot_intensity_distribution,
    plot_intensity_by_sport,
    plot_hr_variability,
    plot_speed_vs_hr
)

def run_categorical_analysis(data_path, output_dir=None):
    """
    Run categorical analysis on workout data.
    
    Args:
        data_path: Path to the workout data file
        output_dir: Directory to save output files (optional)
    
    Returns:
        DataFrame containing processed workout data
    """
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Set up visualization style
    setup_visualization_style()
    
    # Create output directory if it doesn't exist
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_data(data_path)
    
    # Process workouts
    all_workouts = process_workouts(data)
    
    # Convert to DataFrame
    workouts_df = convert_to_dataframe(all_workouts)
    
    # Plot workout counts by sport
    plot_workout_counts_by_sport(workouts_df)
    
    # Plot gender distribution
    plot_gender_distribution(workouts_df)
    
    # Plot heart rate by sport
    sport_colors = plot_heart_rate_by_sport(workouts_df)
    
    # Plot intensity distribution
    plot_intensity_distribution(workouts_df)
    
    # Plot intensity by sport
    sport_counts = workouts_df['sport'].value_counts()
    plot_intensity_by_sport(workouts_df, sport_counts)
    
    # Define top sports
    top_sports = sport_counts[sport_counts >= 50].index.tolist()
    
    # Plot heart rate variability
    plot_hr_variability(workouts_df, top_sports, sport_colors)
    
    # Plot speed vs heart rate
    correlations = plot_speed_vs_hr(workouts_df, top_sports, sport_colors)
    
    return workouts_df

def analyze_workout_intensity(workouts_df):
    """
    Analyze workout intensity metrics.
    
    Args:
        workouts_df: DataFrame containing workout data
        
    Returns:
        DataFrame with extended intensity analysis
    """
    # Define a function to classify workout intensity based on heart rate
    def classify_intensity(avg_hr, max_hr=190):
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
    
    # Add intensity classification to the dataframe if not already present
    if 'intensity' not in workouts_df.columns:
        workouts_df['intensity'] = workouts_df['avg_hr'].apply(classify_intensity)
    
    # Analyze duration vs intensity
    if 'duration' in workouts_df.columns:
        # Convert duration to minutes for better visualization
        workouts_df['duration_min'] = workouts_df['duration'] / 60
        
        # Group by intensity and calculate mean duration
        intensity_duration = workouts_df.groupby('intensity')['duration_min'].agg(['mean', 'median', 'std', 'count'])
        
        # Sort by intensity level
        intensity_order = ['Very Light', 'Light', 'Moderate', 'Hard', 'Very Hard']
        intensity_duration = intensity_duration.reindex(intensity_order)
        
        print("\nAverage workout duration by intensity level:")
        print(intensity_duration)
        
        # Visualize
        plt.figure(figsize=(12, 6))
        bars = plt.bar(intensity_duration.index, intensity_duration['mean'],
                      yerr=intensity_duration['std'], capsize=5,
                      color=['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'])
        
        plt.title('Average Workout Duration by Intensity Level', fontsize=16)
        plt.xlabel('Intensity Level', fontsize=14)
        plt.ylabel('Average Duration (minutes)', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 5,
                   f'n={intensity_duration["count"][bar.get_x() + bar.get_width()/2]:.0f}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    return workouts_df

def analyze_workout_trends(workouts_df):
    """
    Analyze trends in workout data.
    
    Args:
        workouts_df: DataFrame containing workout data
        
    Returns:
        DataFrame with extended trend analysis
    """
    # Check if timestamp or date information is available
    timestamp_cols = [col for col in workouts_df.columns if 'time' in col.lower() or 'date' in col.lower()]
    
    if timestamp_cols and 'sport' in workouts_df.columns:
        timestamp_col = timestamp_cols[0]
        
        # Convert timestamp to datetime if needed
        if pd.api.types.is_numeric_dtype(workouts_df[timestamp_col]):
            workouts_df['date'] = pd.to_datetime(workouts_df[timestamp_col], unit='s')
        else:
            workouts_df['date'] = pd.to_datetime(workouts_df[timestamp_col])
        
        # Extract month and year
        workouts_df['month'] = workouts_df['date'].dt.month
        workouts_df['year'] = workouts_df['date'].dt.year
        
        # Group by month, year, and sport
        monthly_counts = workouts_df.groupby(['year', 'month', 'sport']).size().reset_index(name='count')
        
        # Pivot to get sports as columns
        pivot_counts = monthly_counts.pivot_table(
            index=['year', 'month'],
            columns='sport',
            values='count',
            fill_value=0
        )
        
        # Plot workout counts over time for top sports
        top_sports = workouts_df['sport'].value_counts().nlargest(5).index.tolist()
        
        plt.figure(figsize=(14, 8))
        for sport in top_sports:
            if sport in pivot_counts.columns:
                plt.plot(pivot_counts[sport], marker='o', label=sport)
        
        plt.title('Workout Counts Over Time by Sport', fontsize=16)
        plt.xlabel('Year-Month', fontsize=14)
        plt.ylabel('Number of Workouts', fontsize=14)
        plt.legend(title='Sport Type')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    return workouts_df
