# Heart Rate and Speed Correlation Analysis

## Overview
`Main.py` processes running workout data for both male and female participants, categorizing their maximum speeds and analyzing corresponding heart rate data. The results are plotted through histograms to provide insights into heart rate distribution at different speed ranges with other visualizations provided in `BinsToHeartrate.py` given the data from `Main.py`.

## Prerequisites
### Required Libraries
Ensure the following Python libraries are installed:
- `numpy`
- `matplotlib`
- `Histograms.py` (A custom module for histogram generation)

### Data Files
The script requires two NumPy `.npy` files:
- `male_run_data_with_proper_interpolate.npy`
- `female_run_data_with_proper_interpolate.npy`

These files contain structured datasets from the FitRec project website with the following keys:
- `tar_derived_speed`: An array of speed values recorded during the workout.
- `tar_heart_rate`: An array of corresponding heart rate values.

## Processing Workflow
1. Download the `processed_endomondoHR_proper_interpolate.npy` from `https://sites.google.com/view/fitrec-project/`
2. Load male and female workout data in `DataScrub.py`.
3. Extract speed and heart rate values, ignoring the first and last 10% of data to remove anomalies.
4. Categorize the maximum speed values into predefined bins:
   - `<10 km/h`
   - `10-12 km/h`
   - `12-14 km/h`
   - `14-16 km/h`
   - `>16 km/h`
5. Store corresponding maximum heart rate values in a dictionary for each speed category.
6. Visualize the distribution of heart rates for each speed category using histograms.

## Functions
### `plot_multiple_histograms(speeds_to_hr, gender)`
- Accepts a dictionary mapping speed categories to heart rate values.
- Generates histograms for each speed bin using the `Histograms.meanHistograms()` function.
- Limits the display to a maximum of 5 histograms.

## Output
- Histograms that display the distribution of heart rates across different speed categories for male and female participants.

## Author
[Derek Jensen]

