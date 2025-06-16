import mlflow
import mlflow.xgboost
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import fetch_kddcup99
import warnings
from dotenv import set_key
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

warnings.filterwarnings("ignore")

class Preprocessing:
    def __init__(self, data, target):
        self.categorical_features = None
        self.probe_attacks = [
            "buffer_overflow.",
            "loadmodule.",
            "perl.",
            "neptune.",
            "smurf.",
            "guess_passwd.",
            "pod.",
            "teardrop.",
            "portsweep.",
            "ipsweep.",
            "land.",
            "ftp_write.",
            "back.",
            "imap.",
            "satan.",
            "phf.",
            "nmap.",
            "multihop.",
            "warezmaster.",
            "warezclient.",
            "spy.",
            "rootkit.",
        ]
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.dataset = self.preprocessing_data(data)
        self.target = self.preprocessing_target(target)

    def preprocessing_data(self, data):
        # Convert type data
        X = data.copy()
        for col in X.columns:
            if col not in ["protocol_type", "service", "flag"]:
                X[col] = X[col].astype(np.float32, errors="ignore")

        X["protocol_type"] = X["protocol_type"].str.decode("utf-8")
        X["service"] = X["service"].str.decode("utf-8")
        X["flag"] = X["flag"].str.decode("utf-8")

        # Encode categorical data
        self.categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

        for col in self.categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le

        # Standar Scalar
        self.numerical_features = [
            col for col in X.columns if col not in self.categorical_features
        ]
        X[self.numerical_features] = self.scaler.fit_transform(
            X[self.numerical_features]
        )

        return X

    def preprocessing_target(self, target):
        y = target.copy()

        y = y.str.decode("utf-8")

        y = y.apply(lambda x: 1 if x in self.probe_attacks else 0)

        return y

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
    cv=3,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)

# Fit the model with verbose output
print("Starting Bayesian optimization...")
bayes_search.fit(X_train, y_train)

# Define the model
# model = XGBClassifier()
model = XGBClassifier(**bayes_search.best_params_)
model.fit(X_train, y_train)

# Predict on the test set
print("Predicting on the test set...")
y_pred = model.predict(X_test)

# Start MLflow run
with mlflow.start_run() as run:
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
    mlflow.xgboost.log_model(model, "model", input_example=X_test)

    mlflow.end_run()
