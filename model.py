from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load data
data = load_iris()

# Train and evaluate model using cross-validation
model = LogisticRegression(max_iter=200)

# Perform cross-validation with 5 folds
cv_scores = cross_val_score(model, data.data, data.target, cv=5, scoring='accuracy')

# Print the cross-validation accuracy scores and the average score
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Average accuracy: {cv_scores.mean()}")
