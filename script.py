import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("/app/model.h5")  # Update path for Docker

# Route to check if the server is running
@app.route("/")
def home():
    return "Model is running!"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the input data from the request (assuming JSON data)
        data = request.get_json()

        # Load the dataset from the JSON input
        df = pd.DataFrame(data)

        # Convert timestamp to datetime if applicable
        if 'Timestamp [ms]' in df.columns:
            try:
                df['Timestamp [ms]'] = pd.to_datetime(df['Timestamp [ms]'])
            except Exception as e:
                return jsonify({"error": f"Error parsing timestamps: {e}"}), 400

        # Sort the data by timestamp if applicable
        if 'Timestamp [ms]' in df.columns:
            df = df.sort_values(by='Timestamp [ms]')

        # Select the relevant column for CPU usage
        time_series_data = df[['CPU usage [MHZ]']]

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

        # Return the predictions as JSON
        prediction_results = [{"index": i + sequence_length, "predicted_cpu_usage_mhz": float(pred[0])} 
                              for i, pred in enumerate(predictions_original)]

        return jsonify(prediction_results)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Start the Flask app
if __name__ == "__main__":
    # Use the port from the environment variable
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
