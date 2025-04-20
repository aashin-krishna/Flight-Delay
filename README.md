# Flight Delay Prediction System

This project predicts whether a flight will be delayed by more than 15 minutes using multiple machine learning approaches.

## Features

- Data cleaning and preprocessing pipeline
- Three modeling approaches:
  - Convolutional Neural Network (CNN)
  - Decision Tree Classifier
  - Logistic Regression
- Automated feature engineering
- Consistent preprocessing for training and test data

## Requirements

- Python 3.7+
- Required packages:pandas
numpy
scikit-learn
tensorflow


## Dataset

The system expects two CSV files:
- `flight_delays_train.csv` - Training data with features and target
- `flight_delays_test.csv` - Test data for predictions

## Usage

1. Place your dataset files in the project directory
2. Run the Jupyter notebook `Flight_Delay_Prediction.ipynb`
3. The system will:
 - Clean and preprocess the data
 - Train all three models
 - Generate prediction files:
   - `results_decision_tree.csv`
   - `results_logistic.csv`
   - CNN predictions are handled differently due to the model architecture

## Model Comparison

| Model | Approach | Best For |
|-------|----------|----------|
| CNN | Deep learning with spatial feature extraction | Complex patterns in sequential data |
| Decision Tree | Rule-based classification | Interpretable predictions |
| Logistic Regression | Linear classification | Baseline performance |

## File Structure
flight-delay-prediction/
├── Flight_Delay_Prediction.ipynb # Main notebook
├── README.md # This file
├── cleaned.csv # Cleaned data output
├── df_encoded_train.csv # Preprocessed training data
├── df_encoded_test.csv # Preprocessed test data
├── simple_cnn_model.h5 # Saved CNN model
├── results_decision_tree.csv # Decision Tree predictions
└── results_logistic.csv # Logistic Regression predictions


## Future Improvements

- Add hyperparameter tuning
- Implement ensemble methods
- Add more sophisticated feature engineering
- Include time series analysis for departure times
