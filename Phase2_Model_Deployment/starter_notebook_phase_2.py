from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf  # Ensure consistent TensorFlow imports

# Load the trained model
def load_trained_model(model_path='lstm_model.h5'):
    return load_model(model_path)

# Initialize Flask app
app = Flask(__name__)
model = load_trained_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data from the request
        data = request.get_json(force=True)
        input_data = data['input']  # Expecting a 3D list of shape (samples, time_steps, num_features)

        # Convert input data to a numpy array
        prediction_input = np.array(input_data)

        # Ensure the input shape is correct for batch processing
        predictions = model.predict(prediction_input)

        # Return all predictions as a list
        return jsonify({'prediction': predictions.flatten().tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)

