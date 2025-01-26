from flask import Flask, request, jsonify
import mlflow.sklearn
import numpy as np
import joblib

app = Flask(__name__)

# Load the model from MLflow
# model = mlflow.sklearn.load_model("runs:/c059d03cb74840ec835a78e64775d21e/artifacts/model") #using the RUN id of the best model from the MLFLOW UI.

model_path = "model.pkl"
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)