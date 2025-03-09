import numpy as np
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

def loadFromDataset(save=False):
    """
    Load data from the dataset and save it as numpy arrays
    :param save: if True, save the data as numpy arrays
    :return: bike_data, run_data
    """

    bike_data = []
    run_data = []
    key_list = ['speed', 'sport', 'heart_rate', 'userId', 'gender', 'id', 'altitude', 'timestamp']

    with open('endomondoHR.json') as f:
        for line in f:
            d = json.loads(line.replace("'", "\""))
            if d.get('sport') is None or d.get('speed') is None:
                continue
            if d['sport'] == 'bike':
                bike_data.append({k: d[k] for k in key_list})
            elif d['sport'] == 'run':
                run_data.append({k: d[k] for k in key_list})

    bike_data = np.array(bike_data)
    run_data = np.array(run_data)

    if save:
        np.save('bike_data.npy', bike_data)
        np.save('run_data.npy', run_data)

    return bike_data, run_data

def createDataframe(bike_data):
    """
    Create a pandas dataframe from the numpy data
    """

    # use averages
    bike_df = pd.DataFrame()

    bike_df['average_speed'] = [np.average(x['speed']) for x in bike_data]
    bike_df['userId'] = [x['userId'] for x in bike_data]
    bike_df['average_heart_rate'] = [np.average(x['heart_rate']) for x in bike_data]
    bike_df['gender'] = [x['gender'] for x in bike_data]
    bike_df['average_altitude'] = [np.average(x['altitude']) for x in bike_data]
    bike_df['max_speed'] = [np.max(x['speed']) for x in bike_data]
    bike_df['max_heart_rate'] = [np.max(x['heart_rate']) for x in bike_data]
    bike_df['max_altitude'] = [np.max(x['altitude']) for x in bike_data]
    
    return bike_df

def trainLinearRegressor(X_train, y_train):
    """
    Train a linear regression model
    """

    numeric_features = ['average_speed']
    categorical_features = ['userId', 'gender']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create a pipeline that combines preprocessing and model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Fit the model
    model.fit(X_train, y_train)

    return model

def trainRandomForest(X_train, y_train):
    """
    Train a random forest model
    """

    numeric_features = ['average_speed']
    categorical_features = ['userId', 'gender']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create the pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Define hyperparameter search space
    param_dist = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [10, 20, 30, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }

    # Perform randomized search for hyperparameter tuning
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, 
                                    n_iter=20, cv=5, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # Get the best model
    best_model = random_search.best_estimator_

    return best_model
