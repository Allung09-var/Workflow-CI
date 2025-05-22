import mlflow
from mlflow.data import pandas_dataset
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import fetch_kddcup99
from preprocessing_local import Preprocessing
import warnings
from dotenv import set_key
import os

warnings.filterwarnings("ignore")

# Read and preprocess data
dataset = fetch_kddcup99(as_frame=True)

preprocessing = Preprocessing(dataset.data, dataset.target)
X, y = preprocessing.dataset, preprocessing.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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
    estimator=XGBClassifier(),
    search_spaces=param_search,
    n_iter=32,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)

# Fit the model with verbose output
print("Starting Bayesian optimization...")
bayes_search.fit(X_train, y_train)

# Define the model
print("Create & Trainig Model")
# model = XGBClassifier()
model = XGBClassifier(**bayes_search.best_params_)
model.fit(X_train, y_train)

# Predict on the test set
print("Predicting on the test set...")
y_pred = model.predict(X_test)

input_example = X_test[:10]

# Start MLflow run
with mlflow.start_run() as run:
    mlflow.log_input(
        dataset=pandas_dataset.from_pandas(X_train), context="Training Data"
    )
    mlflow.log_input(dataset=y_train, context="Training Target")

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    set_key(env_path, "MLFLOW_RUN_ID", run.info.run_id)

    # Log the best parameters
    print("Best parameters found:", bayes_search.best_params_)
    mlflow.log_params(bayes_search.best_params_)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Accuracy on the test set: {accuracy}")

    # Log the accuracy
    mlflow.log_metrics(
        {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
        }
    )

    # Log the model
    print("Logging the model to MLflow...")
    mlflow.xgboost.log_model(model, "model", input_example=input_example)

    mlflow.end_run()
