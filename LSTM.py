import pandas as pd
import numpy as np
import pickle as pc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import configparser

# Load the data and drop unnecessary columns
try:
    data = pd.read_csv('btcusd.csv')
    data = data[['Close']]  # Only keep 'Close' column
except FileNotFoundError:
    print("The file 'btcusd.csv' was not found.")
    exit()
except KeyError as e:
    print(f"Missing expected column in data: {e}")
    exit()

# Sample only 1% of the data
data = data.sample(frac=0.1, random_state=0)

# Sort data by index if necessary, as LSTM expects sequential data
data = data.sort_index()

# Normalize the 'Close' values for better performance in LSTM
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences for LSTM input (X) and output (Y)
def create_sequences(data, time_steps=25):
    Xs, Ys = [], []
    for i in range(len(data) - time_steps):
        Xs.append(data[i:i + time_steps])
        Ys.append(data[i + time_steps])
    return np.array(Xs), np.array(Ys)

# Set the time step window (for LSTM)
TIME_STEPS = 25

# Create sequences
X_seq, Y_seq = create_sequences(data_scaled, TIME_STEPS)

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X_seq, Y_seq, test_size=0.3, random_state=0)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=25, return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=25))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(train_x, train_y, epochs=2, batch_size=32, validation_data=(test_x, test_y))

# Save the model
try:
    model.save('LSTM_model_close_only.h5')
    print("LSTM model saved successfully.")
except Exception as e:
    print(f"Error saving the model: {e}")

# Make predictions using the test set
predictions = model.predict(test_x)

# Inverse scale the predictions and actual values to original range
predictions_rescaled = scaler.inverse_transform(predictions)
test_y_rescaled = scaler.inverse_transform(test_y.reshape(-1, 1))


# Evaluate and print the R2 score of the model
print("R2-score: %.2f" % r2_score(test_y_rescaled, predictions_rescaled))
print("MSE-score: %.2f" % mean_squared_error(test_y_rescaled, predictions_rescaled))

# Config dosyasını oluşturma ve okuma
config = configparser.ConfigParser()
config.read('application.properties')

# Yeni bir değer eklemek
config.set('Metrics', 'lstmregression_mse-score',str(mean_squared_error(test_y_rescaled, predictions_rescaled)))

# Dosyayı güncelleme
with open('application.properties', 'w') as configfile:
    config.write(configfile)

# Plot the results
plt.figure(figsize=(14,5))
plt.plot(test_y_rescaled, color='blue', label='Actual Prices')
plt.plot(predictions_rescaled, color='red', label='Predicted Prices')
plt.title('LSTM Model Prediction vs Actual (Close Prices)')
plt.xlabel('Time')
plt.ylabel('BTC Close Price')
plt.legend()
plt.show()
