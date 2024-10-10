# Phase 3: Model Monitoring Using Evidently

## Objective
Set up a monitoring system using Evidently to track data drift and regression performance of the traffic prediction model, with a Flask dashboard and email alerts for real-time monitoring.

### Key Components
- **Evidently**: Used to generate reports for monitoring data drift and regression performance.
- **Flask Web Application**: Serves the monitoring dashboard.
- **Email Alerts**: Notifies the user when data drift exceeds a defined threshold.

### Monitoring Setup Steps
1. **Data Preparation**:
   - Loads training (`X_train`, `y_train`) and test (`X_test`, `y_test`) data from `.npy` files.
   - Creates DataFrames for both datasets, including predictions.

2. **Evidently Report**:
   - Generates a report to monitor data drift and regression performance.
   - Saves the report as `evidently_model_report.html`.

3. **Flask Dashboard**:
   - Run the Flask app to serve the monitoring report on a web page.
   - Access the report via: `http://<server-ip>:5001/monitoring`.

4. **Email Alerts**:
   - Automatically sends email notifications if data drift exceeds the set threshold.

### Running the Monitoring Dashboard
- **Start the Flask App**:
  ```bash
  python phase3_monitoring_script.py
