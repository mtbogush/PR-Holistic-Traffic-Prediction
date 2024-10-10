# Phase 2: Model Deployment Using Kubernetes

## Objective
Deploy the traffic prediction model as a scalable service using Docker and Kubernetes, with a RESTful API for real-time predictions. Test the API using Postman with sample data from the `extracted_data.json` file.

### Key Components
- **Dockerfile**: Defines the environment and dependencies for the application.
- **Kubernetes Deployment**: Uses `deployment.yaml` to manage the pods running the model.
- **Kubernetes Service**: Uses `service.yaml` to expose the model's API to external clients.
- **Flask API**: Provides an endpoint for making traffic predictions using the trained model.

### Deployment Steps
1. **Docker**:
   - Build the Docker image using the provided `Dockerfile`:
     ```bash
     docker build -t mtbogush/model-api:latest .
     ```
   - Push the Docker image to DockerHub:
     ```bash
     docker push mtbogush/model-api:latest
     ```

2. **Kubernetes**:
   - Deploy the model using the `deployment.yaml` file:
     ```bash
     kubectl apply -f deployment.yaml
     ```
   - Expose the model as a service using the `service.yaml` file:
     ```bash
     kubectl apply -f service.yaml
     ```

### API Testing with Postman
- Use Postman to test the API endpoint.
- **Endpoint**: `POST /predict`
- **URL**: `http://<node-ip>:30007/predict`
  - Replace `<node-ip>` with the IP address of your Kubernetes node.
- **Body**: Send a JSON object with the input data in the following format (example truncated to show structure):
  ```json
  {
      "input": [
          [
              64.375, 67.625, 67.125, 61.5, 66.875, 68.75, ...,  # up to 207 values
              70.25, 65.5, 66.25, 68.125, 64.875, 66.0
          ]
      ]
  }
