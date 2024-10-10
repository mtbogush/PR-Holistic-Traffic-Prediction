# Phase 2: Model Deployment Using Kubernetes

## Objective
Deploy the traffic prediction model as a scalable service using Docker and Kubernetes, with a RESTful API for predictions.

### Key Components
- **Dockerfile**: Defines the environment and dependencies for the application.
- **Kubernetes Deployment**: Specifies the configuration for deploying the model on a Kubernetes cluster.
- **Flask API**: Provides an endpoint for making traffic predictions using the trained model.

### Deployment Steps
1. **Docker**: 
   - Build the Docker image using the provided `Dockerfile`.
   - Run: `docker build -t mtbogush/model-api:latest .`
   - Push the image to DockerHub: `docker push mtbogush/model-api:latest`

2. **Kubernetes**:
   - Deploy the service with `traffic-prediction-deployment.yaml`.
   - Use: `kubectl apply -f traffic-prediction-deployment.yaml` to deploy.

### Deliverables
- `Dockerfile`: Configuration for building the model's Docker image.
- `traffic-prediction-deployment.yaml`: YAML file for Kubernetes deployment.
- `api_documentation.pdf`: Documentation for interacting with the model's API.
- `docker_image_link.txt`: Contains the link to the DockerHub image.

### API Endpoint
- **POST /predict**: Accepts JSON input and returns the traffic prediction.
