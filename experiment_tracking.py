import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Set the MLflow experiment name
mlflow.set_experiment("Iris_Model_Experiment")

# 2. Load Iris dataset from CSV
df = pd.read_csv("data/Iris.csv")

# 3. Map species to numeric labels if not already numeric
species_map = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}
df["Species"] = df["Species"].map(species_map)

# 4. Convert all columns to float to avoid MLflow integer-column warnings
df = df.astype(float)

# 5. Shuffle the dataset
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Separate features (X) and target (y)
X = df_shuffled[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
y = df_shuffled["Species"]

# 7. Introduce random noise to reduce the chance of 100% accuracy
np.random.seed(42)
X_noisy = X + np.random.normal(loc=0, scale=0.3, size=X.shape)

# 8. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_noisy,
    y,
    test_size=0.3,
    random_state=42
)

# 9. Define hyperparameters for three runs
hyperparams = [
    {"C": 0.1,  "max_iter": 100},
    {"C": 1.0,  "max_iter": 200},
    {"C": 10.0, "max_iter": 300}
]

for param in hyperparams:
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("C", param["C"])
        mlflow.log_param("max_iter", param["max_iter"])

        # Train the model
        model = LogisticRegression(
            C=param["C"],
            max_iter=param["max_iter"],
            random_state=42
        )
        model.fit(X_train, y_train)

        # Cross-validation accuracy (logged as "accuracy")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mean_cv_accuracy = np.mean(cv_scores)

        # Evaluate on TEST set
        test_preds = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_preds)

        # Precision, recall, and F1 (macro-average for 3 classes)
        precision = precision_score(y_test, test_preds, average="macro")
        recall    = recall_score(y_test, test_preds, average="macro")
        f1        = f1_score(y_test, test_preds, average="macro")

        # Log metrics
        mlflow.log_metric("accuracy", mean_cv_accuracy)   # CROSS-VAL as "accuracy"
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Provide a sample input to avoid feature-name warnings
        sample_input = X_test.iloc[[0]].copy()
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=sample_input
        )

        print(f"\nRun for C={param['C']}, max_iter={param['max_iter']}")
        print(f"Accuracy: {mean_cv_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")

print("\nAll experiments completed!")
