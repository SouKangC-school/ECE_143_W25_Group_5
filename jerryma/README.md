### What I Have Done
1. Analyzed Jerry's personal running data and matched it with Derek's population model.  
2. Examined the relationship between male and female runners' performance and speed ranges, categorizing the population based on athletic ability.  

### Needed Python Packages
- `FitFile`
- `scipy`
- `sklearn`
- `pandas`

### Code Overview
- **`data_split.py`**: Loads personal running data from `.fit` files, selects running-type records, and converts the data format.  
- **`model_plot.py`**:  
  - `basemodel_plot`: Provides a simple data overview.  
  - `runner_indicator_plot`: Analyzes key performance indicators of runners.  
  - `performance_classification_plot`: Uses KMeans clustering to classify runners based on their athletic performance.  

### How to Run
Execute `plot.ipynb` to generate the analysis and visualizations.  
