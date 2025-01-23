import mlflow
import mlflow.sklearn
import optuna
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score

# Enable MLflow Experiment
mlflow.set_experiment("Iris_Model_Hyperparameter_Tuning New")

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define objective function for Optuna
def objective(trial):
    # Define hyperparameters to tune
    max_iter = trial.suggest_int('max_iter', 100, 500)
    C = trial.suggest_float('C', 0.01, 10.0, log=True)
    solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'sag', 'saga'])

    # Start an MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("C", C)
        mlflow.log_param("solver", solver)

        # Create and train the model
        model = LogisticRegression(max_iter=max_iter, C=C, solver=solver, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
        
        predictions = model.predict(X_test)

        # Log metrics
        mlflow.log_metric("accuracy", score)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        print(f"Run complete for max_iter={max_iter}, C={C}, solver={solver}. Accuracy: {score}")

    return score

# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)
print("Best accuracy:", study.best_value)

# Train the final model with the best hyperparameters
best_params = study.best_params
final_model = LogisticRegression(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# Evaluate the final model
predictions = final_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Final Model Accuracy: {accuracy}")


precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')

print(f"Final Model\nPrecision: {precision},\nRecall: {recall}")


# Log the final model with MLflow
with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("final_accuracy", accuracy)
    mlflow.log_metric("final_Precision", precision)
    mlflow.log_metric("final_Recall", recall)
    mlflow.sklearn.log_model(final_model, "final_model")
