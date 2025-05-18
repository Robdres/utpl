import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

database = pd.read_csv("../database/heart.csv")
X_train, X_test, y_train, y_test = train_test_split(
    database[database.columns!=["HeartDisease "]], database["HeartDisease"], test_size=0.3, random_state=42
)
# Set up MLflow experiment (optional)
mlflow.set_experiment("RandomForest_Iris_Classification")

with mlflow.start_run():
    # Hyperparameters
    n_estimators = 100
    max_depth = 3
    random_state = 42

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train model
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    clf.fit(X_train, y_train)

    # Predict and evaluate
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy", acc) #type:ignore

    # Log model
    mlflow.sklearn.log_model(clf, "model")

    print(f"Model logged with accuracy: {acc:.4f}")
