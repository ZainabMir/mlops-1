from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

# Enable MLflow Experiment
mlflow.set_experiment("Iris_Model_Experiment")

# Load dataset
data = load_iris()

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.data)

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

        # Perform cross-validation with 5 folds
        cv_scores = cross_val_score(model, X_scaled, data.target, cv=5, scoring='accuracy')

        # Log metrics (accuracy)
        mlflow.log_metric("accuracy_mean", cv_scores.mean())
        mlflow.log_metric("accuracy_std", cv_scores.std())

        # Log model with input example
        model.fit(X_scaled, data.target)  # Fit the model on the entire dataset
        mlflow.sklearn.log_model(model, "model", input_example=X_scaled[0:1])

        print(f"Run complete for max_iter={param['max_iter']}. "
              f"Mean Accuracy: {cv_scores.mean()} (+/- {cv_scores.std()})")

print("All experiments completed!")
