# Phase 1: Model Experimentation Using Kubeflow

## Objective
This phase involves experimenting with the Linear Regression model using the METR-LA dataset. The primary goal is to build enhanced time-series features, train the model, and evaluate its performance on traffic flow prediction. The experiment is tracked using Kubeflow.

### Tasks
- **Data Preprocessing**: Load and preprocess the data by creating time-based and lagged features.
- **Model Training**: Train a Linear Regression model using the preprocessed features.
- **Model Evaluation**: Evaluate the model's performance using metrics like MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).
- **Visualization**: Plot the actual vs. predicted traffic flow for each sensor location.

### Code Overview

#### Libraries Used
- `pandas` and `numpy`: For data manipulation and analysis.
- `pickle`: To load pre-processed traffic flow data.
- `matplotlib`: For plotting the results.
- `sklearn.linear_model`: For the Linear Regression model.
- `sklearn.metrics`: For evaluation metrics like MAE and RMSE.

#### Data Loading
The data is loaded from a pickle file containing pre-processed traffic flow data for multiple sensor locations.

#### Feature Engineering
The code creates several enhanced features for each sensor location, including:
- **Time-based features**: Hour of the day, day of the week, and weekend indicator.
- **Lagged features**: Traffic flow values lagged by 1, 2, and 3 intervals.
- **Rolling and Exponential Moving Averages**: Smoothing techniques to capture trends.

#### Model Training and Evaluation
- **Linear Regression Model**: The model is trained using an 80/20 train-test split.
- **Evaluation Metrics**: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are used to measure the model's performance.
- **Visualization**: The actual vs. predicted traffic flow is plotted for visual comparison.

### Deliverables
- `model_experimentation.ipynb`: Jupyter notebook containing the code for data preprocessing, feature engineering, model training, evaluation, and visualization.
- `kubeflow_logs.pdf`: Exported logs from Kubeflow tracking the experimentation process.
- `model_comparison_report.pdf`: A report comparing the performance of the Linear Regression model with other models (to be updated later in the project).
- `starter_notebook_phase_1.py`: Starter code for initializing the project.

### How to Run the Experiment
1. **Data Preprocessing**: Run the feature creation function `create_features(location_index)` to generate features for each sensor location.
2. **Model Training**: Use the `train_linear_model(df, location_index)` function to train and evaluate the Linear Regression model.
3. **Visualization**: Visualize the results using the plotted graph of actual vs. predicted values.

### Requirements
- Python libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`
- METR-LA dataset preprocessed into the format provided in the pickle file `tra_Y_tr.pkl`

### Instructions
1. Open the Jupyter notebook `model_experimentation.ipynb`.
2. Run the code cells sequentially to preprocess data, train models, and visualize results.
3. Review the evaluation metrics (MAE and RMSE) and visualizations for model insights.
