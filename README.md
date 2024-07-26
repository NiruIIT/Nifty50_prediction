# Nifty 50 Price Prediction using Bidirectional LSTM

This project aims to predict the Nifty 50 stock prices using a Bidirectional Long Short-Term Memory (LSTM) model. The data used spans from 2015 to 2024 with 5-minute intervals.

## Project Structure

Nifty50_Price_Prediction/
├── nifty50_data.csv
├── nifty50_prediction.py
├── README.md
└── requirements.txt

## Requirements

Before running the code, ensure you have the following packages installed:

- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow

You can install these packages using the following command:

```sh
pip install -r requirements.txt

##Data
The dataset nifty50_data.csv contains Nifty 50 price data with a 'Close' column representing the closing prices.

##Preprocessing
Load and Normalize Data:

Load the data from the CSV file.
Normalize the 'Close' prices to a range between 0 and 1 using MinMaxScaler.
Create Dataset:

Create input-output pairs for the LSTM model.
Use 60 previous time steps to predict the next value.
##Model
A Bidirectional LSTM model is used with the following architecture:

Two Bidirectional LSTM layers with 100 units each.
Dropout layers with a dropout rate of 40% to prevent overfitting.
A Dense layer to produce the final output.
##Training
The model is trained with the following parameters:

Epochs: 200
Batch size: 64
Validation split: 20%
##Evaluation
The model's performance is evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) scores for both training and testing data.

##Future Predictions
The model predicts the Nifty 50 prices for the next 375 minutes (75 5-minute intervals) based on the last test sequence.

##Results
The actual prices, training predictions, testing predictions, and future predictions are plotted for visualization.
