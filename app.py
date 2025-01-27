from flask import Flask, request, jsonify
import mlflow.sklearn
import numpy as np
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the model from a local file using joblib
# Uncomment the following line to load the model from MLflow using the RUN id of the best model from the MLFLOW UI.
# model = mlflow.sklearn.load_model("runs:/c059d03cb74840ec835a78e64775d21e/artifacts/model")

model_path = "model.pkl"
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint that takes a JSON payload with 'features' key,
    performs prediction using the loaded model, and returns the prediction.
    
    Example request payload:
    {
        "features": [value1, value2, value3, ...]
    }
    
    Example response:
    {
        "prediction": predicted_value
    }
    """
    # Get the JSON data from the request
    data = request.get_json()
    
    # Convert the features to a numpy array and reshape for the model
    features = np.array(data['features']).reshape(1, -1)
    
    # Perform the prediction
    prediction = model.predict(features)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    # Run the Flask application on host 0.0.0.0 and port 5000
    app.run(host='0.0.0.0', port=5000)
