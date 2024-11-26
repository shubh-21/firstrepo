import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = tf.keras.models.load_model('/content/model.h5')

# Load the dataset
data = pd.read_csv('/content/dataset.csv')

# Convert timestamp to datetime (if applicable)
if 'Timestamp [ms]' in data.columns:
    try:
        data['Timestamp [ms]'] = pd.to_datetime(data['Timestamp [ms]'])
    except Exception as e:
        print(f"Error parsing timestamps: {e}")

# Sort the data by timestamp (if applicable)
if 'Timestamp [ms]' in data.columns:
    data = data.sort_values(by='Timestamp [ms]')

# Select the relevant columns for CPU usage
time_series_data = data[['CPU usage [MHZ]']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
time_series_data['CPU usage [MHZ]'] = scaler.fit_transform(time_series_data['CPU usage [MHZ]'].values.reshape(-1, 1))

# Prepare data for prediction using the sequence length
sequence_length = 60  # Same as used during training
X = []

for i in range(len(time_series_data) - sequence_length):
    X.append(time_series_data['CPU usage [MHZ]'].values[i:i + sequence_length])

X = np.array(X)

# Reshape X to match the LSTM input shape [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Predict future CPU usage
predictions = model.predict(X)

# Inverse transform the predictions to original scale
predictions_original = scaler.inverse_transform(predictions)

# Print the predictions
for i, pred in enumerate(predictions_original):
    print(f"Predicted CPU usage for sequence ending at index {i + sequence_length}: {pred[0]:.2f} MHz")
