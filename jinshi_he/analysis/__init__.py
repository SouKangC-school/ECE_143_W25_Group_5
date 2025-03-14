"""
Analysis modules for workout data.
"""
from .workout_analysis import run_categorical_analysis, analyze_workout_intensity, analyze_workout_trends
from .heart_rate_analysis import analyze_heart_rate_zones, analyze_training_load, analyze_altitude_impact

__all__ = [
    'run_categorical_analysis',
    'analyze_workout_intensity',
    'analyze_workout_trends',
    'analyze_heart_rate_zones',
    'analyze_training_load',
    'analyze_altitude_impact'
]
