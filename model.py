# model.py with hyperparam tuning

import optuna
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

    # Create model
    model = LogisticRegression(max_iter=max_iter, C=C, solver=solver, random_state=42)

    # Evaluate model using cross-validation
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
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


from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
print(f"Final Model\nPrecision: {precision},\nRecall: {recall}")