"""
Heart rate analysis functionality.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.metrics import hr_zone_distribution, calculate_trimp

def analyze_heart_rate_zones(workouts_df, max_hr=190):
    """
    Analyze heart rate zone distribution in workouts.
    
    Args:
        workouts_df: DataFrame containing workout data
        max_hr: Maximum heart rate to use for zone calculations
        
    Returns:
        DataFrame with heart rate zone analysis
    """
    # Process workouts to get zone distributions
    zone_data = []
    sample_indices = np.random.choice(
        len(workouts_df), 
        min(1000, len(workouts_df)), 
        replace=False
    )

    for idx in sample_indices:
        workout = workouts_df.iloc[idx]
        if 'hr_values' in workout:
            zones = hr_zone_distribution(workout['hr_values'], max_hr)
            zone_data.append({
                'sport': workout['sport'],
                **zones
            })

    zone_df = pd.DataFrame(zone_data)

    # Analyze zone distribution by sport
    top_sports = workouts_df['sport'].value_counts().nlargest(10).index.tolist()
    zone_by_sport = zone_df.groupby('sport').mean()
    zone_by_sport = zone_by_sport.loc[top_sports]  # Filter for top sports

    # Create a stacked bar chart
    plt.figure(figsize=(14, 10))
    zone_cols = [col for col in zone_df.columns if 'Zone' in col]
    zone_by_sport[zone_cols].plot(kind='bar', stacked=True, 
                                 colormap='viridis')

    plt.title('Heart Rate Zone Distribution by Sport', fontsize=16)
    plt.xlabel('Sport Type', fontsize=14)
    plt.ylabel('Percentage of Time in Zone', fontsize=14)
    plt.legend(title='Heart Rate Zone')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return zone_df

def analyze_training_load(workouts_df):
    """
    Analyze training load (TRIMP) for workouts.
    
    Args:
        workouts_df: DataFrame containing workout data
        
    Returns:
        DataFrame with training load analysis
    """
    # Check if required columns exist
    if not all(col in workouts_df.columns for col in ['avg_hr', 'duration']):
        print("Cannot calculate TRIMP: missing required columns (avg_hr, duration)")
        return workouts_df
    
    # Calculate TRIMP for workouts with duration
    trimp_df = workouts_df.dropna(subset=['duration', 'avg_hr']).copy()
    trimp_df['duration_min'] = trimp_df['duration'] / 60
    trimp_df['trimp'] = trimp_df.apply(
        lambda row: calculate_trimp(
            row['avg_hr'], 
            row['duration_min'], 
            row.get('gender', 'male')
        ), 
        axis=1
    )

    # Analyze TRIMP by sport
    trimp_by_sport = trimp_df.groupby('sport')['trimp'].agg(['mean', 'median', 'std', 'count'])
    trimp_by_sport = trimp_by_sport[trimp_by_sport['count'] >= 10].sort_values('mean', ascending=False)

    # Create a bar chart
    plt.figure(figsize=(14, 8))
    sport_colors = sns.color_palette('husl', n_colors=len(trimp_by_sport))
    bars = plt.bar(trimp_by_sport.index, trimp_by_sport['mean'], 
                   yerr=trimp_by_sport['std'], capsize=5, alpha=0.7,
                   color=sport_colors)

    plt.title('Average Training Load (TRIMP) by Sport Type', fontsize=16)
    plt.xlabel('Sport Type', fontsize=14)
    plt.ylabel('Average TRIMP Score', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add count labels
    for i, bar in enumerate(bars):
        count = trimp_by_sport['count'].iloc[i]
        plt.text(bar.get_x() + bar.get_width()/2, 5, 
                 f'n={count}', ha='center', rotation=90, 
                 color='black', fontweight='bold')

    plt.tight_layout()
    plt.show()
    
    # Calculate TRIMP ratio to duration
    trimp_df['trimp_per_min'] = trimp_df['trimp'] / trimp_df['duration_min']
    
    # Plot TRIMP efficiency by sport
    trimp_eff_by_sport = trimp_df.groupby('sport')['trimp_per_min'].agg(['mean', 'std', 'count'])
    trimp_eff_by_sport = trimp_eff_by_sport[trimp_eff_by_sport['count'] >= 10].sort_values('mean', ascending=False)
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(trimp_eff_by_sport.index, trimp_eff_by_sport['mean'], 
                   yerr=trimp_eff_by_sport['std'], capsize=5, alpha=0.7,
                   color=sport_colors[:len(trimp_eff_by_sport)])
    
    plt.title('Training Efficiency (TRIMP per Minute) by Sport Type', fontsize=16)
    plt.xlabel('Sport Type', fontsize=14)
    plt.ylabel('TRIMP per Minute', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return trimp_df

def analyze_altitude_impact(workouts_df):
    """
    Analyze the impact of altitude on heart rate.
    
    Args:
        workouts_df: DataFrame containing workout data
        
    Returns:
        Results of altitude analysis
    """
    # Check if required columns exist
    if not all(col in workouts_df.columns for col in ['total_ascent', 'avg_hr']):
        print("Cannot analyze altitude impact: missing required columns (total_ascent, avg_hr)")
        return None
    
    # Filter for workouts with altitude data
    altitude_df = workouts_df.dropna(subset=['total_ascent', 'avg_hr'])
    
    # Focus on outdoor sports
    outdoor_sports = ['run', 'bike', 'hike', 'walk', 'ski', 'trail']
    
    plt.figure(figsize=(12, 8))
    for sport in outdoor_sports:
        sport_data = altitude_df[altitude_df['sport'] == sport]
        if len(sport_data) > 10:
            plt.scatter(sport_data['total_ascent'], sport_data['avg_hr'], 
                        label=f"{sport} (n={len(sport_data)})", 
                        alpha=0.6)
    
    plt.title('Impact of Elevation Gain on Heart Rate', fontsize=16)
    plt.xlabel('Total Ascent (units)', fontsize=14)
    plt.ylabel('Average Heart Rate (bpm)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Sport Type')
    plt.tight_layout()
    plt.show()
    
    # Calculate correlation between ascent and heart rate for outdoor sports
    correlations = {}
    for sport in outdoor_sports:
        sport_data = altitude_df[altitude_df['sport'] == sport]
        if len(sport_data) > 10:
            corr = sport_data['total_ascent'].corr(sport_data['avg_hr'])
            correlations[sport] = corr
    
    print("Correlation between elevation gain and heart rate:")
    for sport, corr in correlations.items():
        print(f"{sport}: {corr:.3f}")
    
    return correlations
