import mlflow
import mlflow.xgboost
import mlflow.xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# Set MLflow Tracking URI
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Set MLflow Experiment Name
mlflow.set_experiment("Training Model")

# Read and preprocess data
data = pd.read_csv("./Workflow-CI/kddcup99_preprocessing/data.csv")

# Convert data to float
data = data.astype(float)

# Split data into features and target
X = data.drop(
    columns=["binary_label"]
)  # Replace 'target' with the name of your target column
y = data["binary_label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model
model = XGBClassifier(eval_metric="logloss")

# Define the parameter search space
param_search = {
    "n_estimators": (50, 500),
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3, "log-uniform"),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
}

# Perform Bayesian optimization
bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=param_search,
    n_iter=32,
    cv=3,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)

# Start MLflow run
with mlflow.start_run(
    run_name="XGBoost_Bayesian_Optimization", log_system_metrics=True
) as run:
    # Log the parameters
    mlflow.xgboost.autolog()

    # Fit the model with verbose output
    print("Starting Bayesian optimization...")
    bayes_search.fit(X_train, y_train, verbose=1)

    # Log the best parameters
    print("Best parameters found:", bayes_search.best_params_)
    mlflow.log_params(bayes_search.best_params_)

    # Predict on the test set
    print("Predicting on the test set...")
    y_pred = bayes_search.best_estimator_.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the test set: {accuracy}")

    # Log the accuracy
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    print("Logging the model to MLflow...")
    mlflow.xgboost.log_model(bayes_search.best_estimator_, "model")
