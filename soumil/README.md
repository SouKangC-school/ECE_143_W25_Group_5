# Dataset Analysis and Heart Rate Prediction

This project analyzes the dataset to find correlated variables, and tries various models on the dataset for heart rate prediction.

## Overview

* heatmap.py contains files for generating the correlation matrices and models used to train the heart rate predictor.
* main.ipynb contains the matrices and their outputs, along with the trained models and evaluation metrics.

## Models Tested:

1. Linear Regression
2. Random Forest Regression
3. Random Forest Regression with Grid Search for hyperparameter tuning.

## Required Libraries

* numpy
* matplotlib
* scikit-learn
* seaborn (for heatmaps)

## Required Dataset files

The dataset requires the endomondoHR.json file, downloadable from the author's [website](https://sites.google.com/view/fitrec-project/
https://dl.acm.org/doi/fullHtml/10.1145/3308558.3313643)

The Notebook generates `bike_data.npy` and `run_data.npy` after the first run, for faster loading on subsequent runs.

