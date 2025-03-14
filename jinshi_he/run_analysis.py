"""
Main script for running workout data analysis.
"""
import argparse
from pathlib import Path
import pandas as pd

from utils.data_loader import load_data
from utils.preprocessing import process_workouts
from analysis.workout_analysis import run_categorical_analysis, analyze_workout_intensity, analyze_workout_trends
from analysis.heart_rate_analysis import analyze_heart_rate_zones, analyze_training_load, analyze_altitude_impact
import config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run workout data analysis')
    
    # Data parameters
    parser.add_argument('--data', type=str, default=config.DATA_PATH,
                        help='Path to data file')
    
    # Analysis parameters
    parser.add_argument('--analysis', type=str, nargs='+', 
                        default=['categorical', 'intensity', 'trends', 'zones', 'trimp', 'altitude'],
                        choices=['categorical', 'intensity', 'trends', 'zones', 'trimp', 'altitude', 'all'],
                        help='Analyses to run')
    
    # Output parameters
    parser.add_argument('--output', type=str, default='analysis_output',
                        help='Output directory for analysis results')
    
    # Save parameters
    parser.add_argument('--save_df', action='store_true',
                        help='Save processed DataFrame to file')
    
    return parser.parse_args()

def main():
    """Main function for running the analysis."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If 'all' is specified, run all analyses
    if 'all' in args.analysis:
        args.analysis = ['categorical', 'intensity', 'trends', 'zones', 'trimp', 'altitude']
    
    # Run categorical analysis (this is the base analysis that creates the DataFrame)
    if 'categorical' in args.analysis:
        print("Running categorical analysis...")
        workouts_df = run_categorical_analysis(args.data, output_dir)
    else:
        # If not running categorical analysis, still need to load and process data
        print("Loading and processing data...")
        data = load_data(args.data)
        all_workouts = process_workouts(data)
        workouts_df = pd.DataFrame(all_workouts)
    
    # Run additional analyses if specified
    if 'intensity' in args.analysis:
        print("\nRunning workout intensity analysis...")
        workouts_df = analyze_workout_intensity(workouts_df)
    
    if 'trends' in args.analysis:
        print("\nRunning workout trends analysis...")
        workouts_df = analyze_workout_trends(workouts_df)
    
    if 'zones' in args.analysis:
        print("\nRunning heart rate zones analysis...")
        zone_df = analyze_heart_rate_zones(workouts_df)
        
        if args.save_df:
            zone_df.to_csv(output_dir / 'heart_rate_zones.csv', index=False)
    
    if 'trimp' in args.analysis:
        print("\nRunning training load (TRIMP) analysis...")
        trimp_df = analyze_training_load(workouts_df)
        
        if args.save_df and trimp_df is not None:
            trimp_df.to_csv(output_dir / 'trimp_analysis.csv', index=False)
    
    if 'altitude' in args.analysis:
        print("\nRunning altitude impact analysis...")
        altitude_correlations = analyze_altitude_impact(workouts_df)
        
        if args.save_df and altitude_correlations is not None:
            # Save correlations to a text file
            with open(output_dir / 'altitude_correlations.txt', 'w') as f:
                f.write("Correlation between elevation gain and heart rate:\n")
                for sport, corr in altitude_correlations.items():
                    f.write(f"{sport}: {corr:.3f}\n")
    
    # Save processed DataFrame if requested
    if args.save_df:
        workouts_df.to_csv(output_dir / 'processed_workouts.csv', index=False)
        print(f"\nProcessed workout data saved to {output_dir / 'processed_workouts.csv'}")

if __name__ == "__main__":
    main()
