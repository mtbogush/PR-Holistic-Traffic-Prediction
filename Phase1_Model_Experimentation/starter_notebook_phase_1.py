import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import logging
import h5py
import tensorflow as tf

# Step 1: Set up logging to a file (train.log)
logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Step 2: Load and preprocess the dataset
def load_data():
    with h5py.File(r'C:\foai\94879-starter-code-Team-Project\metr-la.h5', 'r') as f:
        df_group = f['df']
        
        # Load the column names (features) from 'block0_items'
        columns = list(df_group['block0_items'][:].astype(str))
        
        # Load the actual data from 'block0_values'
        data = df_group['block0_values'][:]
        
        # Convert the data into a DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        return df

def preprocess_data(data):
    # Use MinMaxScaler to normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

# Step 3: Create sequences for time-series forecasting with 12 time steps for both training and prediction
def create_sequences(data, time_steps=12):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])  # Sequence of features for input
        y.append(data[i + time_steps, 0])  # Predict the next value for the first feature
    return np.array(X), np.array(y)

# Step 4: Define the LSTM and GRU models with 12 units in the output layer for predicting 12 steps
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))  # Single unit as we predict only the next time step
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model

def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(64, input_shape=input_shape, return_sequences=True))
    model.add(GRU(32))
    model.add(Dense(1))  # Single unit as we predict only the next time step
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model

# Step 5: Train and evaluate the model, and log to MLflow
def train_and_evaluate(model_type="LSTM"):
    # Load and preprocess the data
    data = load_data()
    data_scaled, scaler = preprocess_data(data)

    # Create sequences with 12 time steps for both training and prediction
    X, y = create_sequences(data_scaled, time_steps=12)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check the number of features in the dataset
    num_features = X_train.shape[2] if len(X_train.shape) > 2 else 1

    # Reshape the data to (samples, time_steps, num_features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], num_features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], num_features))
    
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Start an MLflow run
    with mlflow.start_run():
        # Select and build the model
        if model_type == "LSTM":
            model = build_lstm_model(input_shape)
            logging.info(f"Building LSTM model with input shape {input_shape}")
        elif model_type == "GRU":
            model = build_gru_model(input_shape)
            logging.info(f"Building GRU model with input shape {input_shape}")

        # Log the model type as a parameter
        mlflow.log_param("model_type", model_type)
        logging.info(f"Starting training for {model_type} model...")

        # Train the model
        history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

        # Make predictions on training data
        y_pred_train = model.predict(X_train)

        # Make predictions on test data
        y_pred_test = model.predict(X_test)

        # Calculate metrics for test data
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        # Log metrics
        mlflow.log_metric("mae_test", mae_test)
        mlflow.log_metric("mse_test", mse_test)
        mlflow.log_metric("r2_score_test", r2_test)

        logging.info(f"Test metrics - MAE: {mae_test}, MSE: {mse_test}, R2: {r2_test}")

        # Save training and test data, including predictions
        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)
        np.save('y_pred_train.npy', y_pred_train)  # Save training predictions
        np.save('X_test.npy', X_test)
        np.save('y_test.npy', y_test)
        np.save(f'y_pred_test_{model_type}.npy', y_pred_test)  # Save test predictions

        # Log the trained model to MLflow
        mlflow.keras.log_model(model, "traffic_prediction_model")

        # Save the model locally as 'lstm_model.h5'
        model.save(f'{model_type.lower()}_model.h5')

        # Upload the train.log file as an artifact
        mlflow.log_artifact("train.log")

        logging.info(f"{model_type} Model training complete. Test MAE: {mae_test}, MSE: {mse_test}, R2 Score: {r2_test}")

if __name__ == "__main__":
    # Example of running the training for LSTM
    train_and_evaluate(model_type="LSTM")
