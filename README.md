# NYC Taxi Fare Prediction

A machine learning model to predict New York City taxi fares using RandomForest regression with advanced feature engineering.

## Overview

This project builds a taxi fare prediction model using historical NYC taxi ride data. The model achieves a Mean Absolute Error (MAE) of approximately $2.19 on validation data, providing accurate fare estimates for taxi rides.

## Dataset
[Dataset](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data)

## Features

- **Distance Calculations**: Haversine distance, Manhattan distance, and bearing between pickup and dropoff locations
- **Time-based Features**: Circular encoding of hour and day of week to capture temporal patterns
- **Advanced Preprocessing**: Data type optimization, outlier removal, and feature scaling
- **Model Optimization**: Hyperparameter tuning with parallel processing

## Model Architecture

- **Algorithm**: RandomForest Regressor
- **Key Parameters**: 
  - max_depth: 10
  - n_estimators: 100
  - n_jobs: 4 (parallel processing)
  - max_samples: 1,000,000

## Performance

- **Training MAE**: 2.18
- **Validation MAE**: 2.19
- **Dataset Size**: 55+ million taxi rides

## Features Used

1. **Pickup/Dropoff Coordinates**: Latitude and longitude
2. **Distance Metrics**: Haversine and Manhattan distances
3. **Temporal Features**: Hour (sin/cos), day of week (sin/cos), month
4. **Ride Details**: Passenger count
5. **Derived Features**: Weekend indicator, bearing angle

## Usage

```bash
python main.py
```

The script will:
1. Load and preprocess the training data
2. Engineer features with distance calculations
3. Train the RandomForest model
4. Generate predictions for test data
5. Save results to submission.csv

## Requirements

- pandas
- scikit-learn
- numpy

## Data

The model expects two CSV files:
- `train.csv`: Training data with fare_amount target
- `test.csv`: Test data for predictions

## Output

- `submission.csv`: Predictions for test data
- `taxi_fare_model.pkl`: Trained model (saved automatically)

## Performance Notes

The model uses vectorized operations for efficient processing of large datasets and includes timing information for each processing step.
