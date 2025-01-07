import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Enable MLflow Experiment
mlflow.set_experiment("Iris_Model_Experiment")

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Different hyperparameters to experiment with
params = [
    {'max_iter': 100},
    {'max_iter': 200},
    {'max_iter': 300}
]

# Loop through different models and track experiments
for param in params:
    with mlflow.start_run():
        # Log parameter
        mlflow.log_param("max_iter", param['max_iter'])

        # Train model
        model = LogisticRegression(max_iter=param['max_iter'])
        model.fit(X_train, y_train)

        # Predict and calculate accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"Run complete for max_iter={param['max_iter']}. Accuracy: {accuracy}")

print("All experiments completed!")
