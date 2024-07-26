import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load your Nifty 50 data (assuming it's in a CSV file named 'nifty50_data.csv')
nifty_data = pd.read_excel("D:\\OPTION_DATA\\NIFTY50.xlsx")

# Preprocess the data
# Assuming 'Close' prices are used for prediction
data = nifty_data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare data for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60  # Number of previous time steps to use for prediction
X, y = create_dataset(scaled_data, time_step)

# Reshape data for LSTM (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Build Bidirectional LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(time_step, 1))))
model.add(Dropout(0.4))  # Increased dropout rate
model.add(Bidirectional(LSTM(units=100)))
model.add(Dropout(0.4))  # Increased dropout rate
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

# Predictions on training and testing data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions to get actual prices
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Calculate performance metrics
train_mse = mean_squared_error(y_train[0], train_predict[:, 0])
test_mse = mean_squared_error(y_test[0], test_predict[:, 0])
train_mae = mean_absolute_error(y_train[0], train_predict[:, 0])
test_mae = mean_absolute_error(y_test[0], test_predict[:, 0])
train_r2 = r2_score(y_train[0], train_predict[:, 0])
test_r2 = r2_score(y_test[0], test_predict[:, 0])

print(f'Train MSE: {train_mse:.2f}')
print(f'Test MSE: {test_mse:.2f}')
print(f'Train MAE: {train_mae:.2f}')
print(f'Test MAE: {test_mae:.2f}')
print(f'Train R2: {train_r2:.2f}')
print(f'Test R2: {test_r2:.2f}')

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(np.arange(0, len(data)), data.flatten(), color='red', label='Actual Nifty 50 Prices')
plt.plot(np.arange(time_step, time_step + len(train_predict)), train_predict.flatten(), color='blue', label='Train Predictions')
plt.plot(np.arange(len(data) - len(test_predict), len(data)), test_predict.flatten(), color='green', label='Test Predictions')
plt.title('Nifty 50 Price Prediction using Bidirectional LSTM')
plt.xlabel('Time')
plt.ylabel('Nifty 50 Price')
plt.legend()
plt.show()

# Predict future prices for the next 375 minutes (75 5-minute intervals)
future_steps = 75
x_input = X_test[-1]  # Use the last test sequence as the starting point
temp_input = list(x_input.flatten())

# Predict future prices
future_predictions = []
for i in range(future_steps):
    if len(temp_input) >= time_step:
        x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
        y_pred = model.predict(x_input, verbose=0)
        future_predictions.append(y_pred[0][0])
        temp_input.append(y_pred[0][0])
    else:
        break

# Invert future predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Plotting future predictions
plt.figure(figsize=(12, 6))
plt.plot(np.arange(0, len(data)), data.flatten(), color='red', label='Actual Nifty 50 Prices')
plt.plot(np.arange(len(data), len(data) + future_steps), future_predictions.flatten(), color='purple', label='Future Predictions')
plt.title('Future Nifty 50 Price Prediction for Next 375 Minutes')
plt.xlabel('Time (5-minute intervals)')
plt.ylabel('Nifty 50 Price')
plt.legend()
plt.show()
