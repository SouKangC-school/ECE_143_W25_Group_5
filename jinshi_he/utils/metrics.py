"""
Metrics utilities for model evaluation.
"""
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for regression.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

def calculate_trimp(avg_hr, duration_min, gender='male', max_hr=190, rest_hr=60):
    """
    Calculate Training Impulse (TRIMP) score.
    
    Args:
        avg_hr: Average heart rate
        duration_min: Duration in minutes
        gender: Gender ('male' or 'female')
        max_hr: Maximum heart rate
        rest_hr: Resting heart rate
        
    Returns:
        TRIMP score
    """
    # Heart rate reserve (HRR)
    hrr = (avg_hr - rest_hr) / (max_hr - rest_hr)
    
    # Gender-specific factor
    if gender.lower() == 'female':
        k = 1.67
    else:
        k = 1.92
    
    # TRIMP formula
    trimp = duration_min * hrr * 0.64 * math.exp(k * hrr)
    return trimp

def hr_zone_distribution(hr_values, max_hr=190):
    """
    Calculate percentage of time spent in each heart rate zone.
    
    Args:
        hr_values: List of heart rate values
        max_hr: Maximum heart rate
        
    Returns:
        Dictionary of zone percentages
    """
    zones = {
        'Zone 1 (50-60%)': 0,
        'Zone 2 (60-70%)': 0,
        'Zone 3 (70-80%)': 0,
        'Zone 4 (80-90%)': 0,
        'Zone 5 (90-100%)': 0
    }
    
    for hr in hr_values:
        hr_percent = hr / max_hr * 100
        if hr_percent < 60:
            zones['Zone 1 (50-60%)'] += 1
        elif hr_percent < 70:
            zones['Zone 2 (60-70%)'] += 1
        elif hr_percent < 80:
            zones['Zone 3 (70-80%)'] += 1
        elif hr_percent < 90:
            zones['Zone 4 (80-90%)'] += 1
        else:
            zones['Zone 5 (90-100%)'] += 1
    
    total = sum(zones.values())
    if total > 0:
        return {zone: count / total * 100 for zone, count in zones.items()}
    else:
        return zones
