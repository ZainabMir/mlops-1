# **MLOps Assignment - Task M3: Model Experimentation and Packaging**

This README provides a detailed guide for running **Task M3** of the MLOps assignment, which involves hyperparameter tuning, model packaging, and deploying the model using Docker and Flask. The file structure and step-by-step instructions are provided below.

---

## **File Structure**

The repository contains the following files and directories:

```
mtruns/                # MLflow tracking directory (auto-generated by MLflow)
.gitignore             # Specifies files to ignore in Git
app.py                 # Flask application for serving the model
Dockerfile             # Dockerfile for containerizing the Flask app
experiment_tracking.py # Script for tracking experiments using MLflow
model.pkl              # Serialized machine learning model
model.py               # Script for training and tuning the model
requirements.txt       # Python dependencies for the project
```

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
