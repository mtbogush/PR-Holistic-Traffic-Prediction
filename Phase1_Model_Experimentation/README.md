# Phase 1: Model Experimentation Using Kubeflow

## Objective
This phase focuses on experimenting with LSTM and GRU models for time-series forecasting using the METR-LA dataset. The experiment includes data preprocessing, sequence creation, model training, evaluation, and tracking with MLflow.

### Tasks
- **Data Loading**: Load the METR-LA dataset using the `h5py` library.
- **Data Preprocessing**: Normalize the data using `MinMaxScaler`.
- **Sequence Creation**: Generate sequences for time-series forecasting.
- **Model Building**: Implement LSTM and GRU models using TensorFlow and Keras.
- **Model Evaluation**: Evaluate model performance using metrics such as MAE (Mean Absolute Error), MSE (Mean Squared Error), and R² Score.
- **Logging and Tracking**: Log training metrics and artifacts using MLflow for experiment tracking.

### Code Overview

#### Libraries Used
- `numpy` and `pandas`: For data manipulation and numerical operations.
- `scikit-learn`: For scaling data and evaluation metrics.
- `h5py`: For handling the METR-LA dataset.
- `tensorflow.keras`: For building LSTM and GRU models.
- `mlflow`: For tracking model training, parameters, and metrics.
- `logging`: For logging training details to a file.

#### Data Loading
The code loads the METR-LA dataset using the `h5py` library, converts it into a DataFrame, and prepares it for further preprocessing.

#### Data Preprocessing
- **Min-Max Normalization**: The data is scaled using `MinMaxScaler` to ensure that all features have values between 0 and 1.

#### Model Architecture
- **LSTM Model**: Composed of two LSTM layers followed by a dense layer for output.
- **GRU Model**: Composed of two GRU layers followed by a dense layer for output.

#### Model Training and Evaluation
- The code trains the selected model (LSTM or GRU) using a defined sequence length.
- Model evaluation metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.
- The training process and results are logged to MLflow, and the model is saved locally.

#### Logging and Experiment Tracking
- **MLflow**: Used to track experiment parameters, metrics, and models.
- **Logging**: Outputs training details to a `train.log` file for analysis.

### Deliverables
- `model_experimentation.ipynb`: Jupyter notebook containing code for data preprocessing, sequence creation, model training, and evaluation.
- `train.log`: Log file detailing the training process.
- `kubeflow_logs.pdf`: (To be added) Exported logs from Kubeflow tracking the experimentation process.
- `model_comparison_report.pdf`: (To be added) Report comparing LSTM and GRU models.
- `starter_notebook_phase_1.py`: Starter code for initializing the project.

### How to Run the Experiment
1. **Data Loading**: Run the `load_data()` function to load the METR-LA dataset.
2. **Data Preprocessing**: Use the `preprocess_data(data)` function to normalize the dataset.
3. **Sequence Creation**: Call the `create_sequences(data)` function to generate sequences for model training.
4. **Model Training**: Use the `train_and_evaluate(model_type="LSTM")` function to train the model (LSTM or GRU).
5. **Experiment Tracking**: View model performance metrics and logs using MLflow.

### Requirements
- Python libraries: `numpy`, `pandas`, `scikit-learn`, `h5py`, `tensorflow`, `mlflow`, `matplotlib`
- METR-LA dataset file: `metr-la.h5` located at `C:\foai\94879-starter-code-Team-Project\metr-la.h5`

### Instructions
1. Open the Jupyter notebook `model_experimentation.ipynb`.
2. Follow the instructions in the notebook to load data, preprocess it, create sequences, and train models.
3. Monitor the model performance through MLflow tracking and review the log file `train.log` for detailed analysis.
