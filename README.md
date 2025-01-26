

# MLOps Foundations

## Overview
This repository demonstrates foundational MLOps practices by implementing a CI/CD pipeline and experiment tracking for a sample machine learning project. The project is divided into three modules:

1. **M1: MLOPS Foundations**
2. **M2: Process and Tooling**
3. **M2: Model Experimentation and Packaging**


## **File Structure**

The repository contains the following files and directories:

```
data/                  # contains Iris.csv used in the model
mlruns/                # MLflow tracking directory (auto-generated by MLflow)
.gitignore             # Specifies files to ignore in Git
app.py                 # Flask application for serving the model
Dockerfile             # Dockerfile for containerizing the Flask app
experiment_tracking.py # Script for tracking experiments using MLflow
model.pkl              # Serialized machine learning model
model.py               # Script for training and tuning the model
requirements.txt       # Python dependencies for the project
README.md              # Project documentation
```

---
---

## M1: 
## Task 1 : CI/CD Pipeline

### Objective
Automate code testing, linting, and deployment for a machine learning model using GitHub Actions.

### Pipeline Details

#### **Trigger Events**
- **Push Events**: Any code changes pushed to the main branch.
- **Pull Requests**: Any pull request targeting the main branch.

#### **Workflow File Path**
`.github/workflows/ci-cd-pipeline.yml`

#### **Pipeline Stages**
1. **Checkout Code**:
   - Fetches the latest code from the repository to ensure the pipeline uses the most recent version.

2. **Set Up Python Environment**:
   - Installs Python 3.9 and prepares the environment.

3. **Install Dependencies**:
   - Installs required libraries using the `requirements.txt` file.

4. **Lint Code**:
   - Checks code formatting and potential issues using `flake8`.

5. **Test Model Script**:
   - Runs the `model.py` script to verify successful execution.

6. **Deploy (Dummy Step)**:
   - Placeholder for a future deployment step.

### Results
- The pipeline runs successfully on both push events and pull requests.
- Logs for each stage were verified, showing no errors.
- Successful execution is marked by green checkmarks in the GitHub Actions CI/CD dashboard.

### Screenshots
1. **GitHub Actions Overview**
   - Displays successful CI/CD runs with green statuses.
   - Red statuses for earlier runs (due to dependency conflicts) were resolved.

2. **Detailed Logs for Each Stage**
   - Verified logs for each pipeline stage, showing linting, testing, and other steps completed successfully.

### Version Control
- Feature branches were used for task implementation:
  - Example: `git checkout -b feature/model-update`
- Changes were pushed to feature branches:
  - Example: `git push origin feature/model-update`

---

## M2: Experiment Tracking with MLflow

### Objective
Track and log machine learning experiments, including metrics, parameters, and artifacts, using MLflow.

### Steps

#### **Step 1: Dataset Preparation**
- **Dataset**: Iris dataset.
- **Split**: 80-20 ratio for training and testing sets.

#### **Step 2: Experiment Tracking Setup**
- **MLflow Tracking Server**: Configured locally to track and log experiments.
- **Experiment Name**: `Iris_Model_Experiment`

#### **Step 3: Model Training and Logging**
- **Algorithm Used**: Logistic Regression.
- **Hyperparameters Tested**:
  - `max_iter`: 100, 200, 300.
  - `C`: Regularization strength.
- **Metrics Logged**:
  - `accuracy`: Average cross-validation accuracy.
  - `test_accuracy`: Accuracy on the test set.
  - `precision`, `recall`, and `f1_score`: Macro-averaged scores on the test set.
- **Artifacts Logged**:
  - Trained models were stored as artifacts for reproducibility.

#### **Step 4: Script Execution**
- Script executed locally using:
  ```bash
  python experiment_tracking.py
  ```
- Three model runs were logged with varying hyperparameters.

#### **Step 5: MLflow UI**
- Started the MLflow UI locally using:
  ```bash
  mlflow ui
  ```
- Verified logs, metrics, parameters, and stored models for all runs at `http://localhost:5000`.

---

## How to Use

### Setting Up the CI/CD Pipeline
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the repository directory:
   ```bash
   cd <repository-name>
   ```
3. Push changes to the `main` branch or create a pull request to trigger the pipeline.

### Running Experiment Tracking
1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute the experiment script:
   ```bash
   python experiment_tracking.py
   ```
3. Start the MLflow UI:
   ```bash
   mlflow ui
   ```
4. Open `http://localhost:5000` in the browser to explore the experiment logs.

---





## Task M3: Model Experimentation and Packaging**

**Task M3** involves hyperparameter tuning, model packaging, and deploying the model using Docker and Flask. The file structure and step-by-step instructions are provided below.

---


## **Step-by-Step Process for Running Task M3**

### **1. Hyperparameter Tuning of Model**

#### **Step 1: Create a Conda Environment**
Create a new Conda environment to isolate dependencies:
```bash
conda create -n mlops-m3 python=3.9
```

#### **Step 2: Activate the Environment**
Activate the Conda environment:
```bash
conda activate mlops-m3
```

#### **Step 3: Install Requirements**
Install the required Python packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### **Step 4: Run `model.py` and MLflow**
- Open two terminals.
- In the first terminal, run `model.py` to perform hyperparameter tuning and log experiments using MLflow:
  ```bash
  python model.py
  ```
- In the second terminal, start the MLflow UI to track experiments:
  ```bash
  mlflow ui
  ```
  - Open your browser and navigate to `http://localhost:5000` to view the MLflow UI.

---

### **2. Docker Running and Hitting API**

#### **Step 1: Build the Docker Image**
Build the Docker image for the Flask application:
```bash
sudo docker build -t ml-model-app .
```

#### **Step 2: Run the Docker Container**
Run the Docker container and map port `5000` (inside the container) to port `6000` (on your host machine):
```bash
sudo docker run -p 6000:5000 ml-model-app
```

#### **Step 3: Test the API**
Once the Docker container is running, use an API testing tool (e.g., Postman, `curl`, or a browser) to send a POST request to the API endpoint.

- **API Endpoint**: `http://172.17.0.2:5000/predict`
- **Request Method**: `POST`
- **Request Body** (JSON):
  ```json
  {
    "features": [1.1, 3.5, 1.9, 6.9]
  }
  ```

- **Example `curl` Command**:
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"features": [1.1, 3.5, 1.9, 6.9]}' http://172.17.0.2:5000/predict
  ```

- **Expected Response**:
  ```json
  {
    "prediction": 2
  }
  ```

---

## **Detailed Explanation of Files**

### **1. `model.py`**
- This script trains a `LogisticRegression` model on the Iris dataset.
- It performs hyperparameter tuning using **Optuna** and logs experiments using **MLflow**.
- The best model is saved as `model.pkl`.

### **2. `experiment_tracking.py`**
- This script demonstrates experiment tracking using **MLflow**.
- It trains multiple models with different hyperparameters and logs the results.

### **3. `app.py`**
- This is a Flask application that serves the trained model.
- It loads the `model.pkl` file and provides an API endpoint (`/predict`) for making predictions.

### **4. `Dockerfile`**
- This file defines the Docker image for the Flask application.
- It installs dependencies, copies the application code, and exposes port `5000`.

### **5. `requirements.txt`**
- Lists all Python dependencies required for the project, including:
  - `Flask`
  - `scikit-learn`
  - `mlflow`
  - `optuna`
  - `joblib`
  - `numpy`

### **6. `mlruns/`**
- This directory is auto-generated by **MLflow** to store experiment logs and artifacts.

---
