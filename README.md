# ECE_143_W25_Group_5 

This repository contains a comprehensive analysis of workout data, focusing on heart rate patterns, speed correlations, and predictive modeling for fitness applications. Our team has explored various aspects of the FitRec dataset to derive meaningful insights and develop predictive models.

## Dataset

This project utilizes the **FitRec** dataset, which contains workout data from the Endomondo fitness tracking application. The dataset includes time-series workout information such as:
- Heart rate measurements
- Speed/pace data
- Distance
- Altitude
- User demographics (gender)
- Sport type

The dataset can be downloaded from the [FitRec Project website](https://sites.google.com/view/fitrec-project/).

Original research paper: [FitRec: A Privacy-Preserving Framework for Fitness Recommendation](https://dl.acm.org/doi/fullHtml/10.1145/3308558.3313643)

## Repository Structure

The repository is organized into four main directories, each containing work from a different team member:

```
repository/
├── derek/                    # Derek Jensen's work
│   ├── BinsToHeartrate.py
│   ├── DataScrub.py
│   ├── Histograms.py
│   ├── Main.py
│   ├── Plots.ipynb
│   └── README.md
├── jerryma/                  # Jerry Ma's work  
│   ├── data_load.py
│   ├── model_plot.py
│   ├── plot.ipynb
│   └── README.md
├── jinshi_he/                # Jinshi He's work
│   ├── analysis/
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   ├── training/
│   ├── utils/
│   ├── config.py
│   ├── README.md
│   ├── run_analysis.py
│   └── train_model.py
├── soumil/                   # Soumil Paranjpay's work
│   ├── heatmap.py
│   ├── main.ipynb
│   └── README.md
├── group_5_notebooks.ipynb   # Combined notebook with all team members' work
└── README.md                 # This file
```

## Team Members and Contributions

### Derek Jensen
**Focus Area**: Heart Rate and Speed Correlation Analysis

Derek's work focuses on analyzing the relationship between maximum speeds and heart rates for both male and female runners. His code categorizes workout data into speed bins and visualizes heart rate distributions for each category.

**Key Files**:
- `Main.py`: Processes workout data and performs binning by speed
- `BinsToHeartrate.py`: Generates visualizations for heart rate data
- `Histograms.py`: Custom module for histogram generation

### Jerry Ma
**Focus Area**: Personal Data Comparison and Performance Classification

Jerry's contribution involves analyzing personal running data against the population model and examining performance relationships between male and female runners.

**Key Files**:
- `data_load.py`: Loads and converts personal running data
- `model_plot.py`: Contains visualization functions for data overview and performance analysis
- `plot.ipynb`: Main notebook for generating analysis and visualizations

### Jinshi He
**Focus Area**: Categorical Analysis and Heart Rate Prediction Models

Jinshi's work includes comprehensive workout data analysis and implementation of various time-series models for heart rate prediction.

**Project Structure**:
- Notebooks for categorical data analysis and LSTM modeling
- Modular code structure for data analysis and model training
- Multiple model architectures including LSTM, RNN, CNN, and Transformer models

### Soumil Paranjpay
**Focus Area**: Dataset Correlation Analysis and Heart Rate Prediction Models

Soumil's contribution involves identifying correlated variables in the dataset and experimenting with different regression models for heart rate prediction.

**Key Files**:
- `heatmap.py`: Contains functions for generating correlation matrices and training models
- `main.ipynb`: Displays matrices, trained models, and evaluation metrics

## Getting Started

### Prerequisites

- Python 3.7+
- The main libraries required across all components:
  - NumPy
  - Pandas
  - Matplotlib
  - scikit-learn
  - PyTorch (for deep learning models)
  - Seaborn (for visualizations)
  - FitFile (for parsing .fit files)
  - Jupyter (for notebooks)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/SouKangC-school/ECE_143_W25_Group_5.git
   cd ECE_143_W25_Group_5
   ```

2. Install the required dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn torch seaborn jupyter fitparse
   ```

3. Download the FitRec dataset from [their website](https://sites.google.com/view/fitrec-project/) and place the files in the appropriate locations as indicated in each component's documentation.

## Running the Analysis

Each team member's directory contains specific instructions for running their portion of the analysis. Here's a quick overview:

### Derek's Analysis
```bash
cd derek
jupyter notebook Plots.ipynb
```

### Jerry's Analysis
```bash
cd jerryma
jupyter notebook plot.ipynb
```

### Jinshi's Analysis
```bash
cd jinshi_he
# For jupyter notebooks
jupyter notebook notebooks/categorical_data_analysis.ipynb
jupyter notebook notebooks/LSTM_model.ipynb

# For categorical analysis:
python run_analysis.py --data path/to/data_shrunk.npy --analysis categorical

# For model training:
python train_model.py --model lstm --data path/to/data_shrunk.npy
```

### Soumil's Analysis
```bash
cd soumil
jupyter notebook main.ipynb
```

### Combined Notebook
```bash
jupyter notebook group_5_notebooks.ipynb
```

The `group_5_notebooks.ipynb` file contains the combined work from all team members in a single notebook, providing an overview of the entire project's analysis and results.

## Results and Findings

This project provides several key insights:
- Correlation patterns between heart rate and speed during workouts
- Performance classification of runners based on their data
- Effective models for heart rate prediction using time-series data
- Gender-based differences in workout performance metrics

## License

This project is licensed under the MIT License.

## Acknowledgements

- The FitRec Project for providing the dataset
- Endomondo users who contributed their workout data