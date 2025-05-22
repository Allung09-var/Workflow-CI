from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

import warnings

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
