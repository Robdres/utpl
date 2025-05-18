#This script is called to train the database again
import sys
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os
load_dotenv()

FLASK_PORT=os.getenv('FLASK_PORT')
EXPERIMENT_NAME=os.getenv('EXPERIMENT_NAME')
MLFLOW_PORT=os.getenv('MLFLOW_PORT')
DATABASE_PATH=os.getenv('DATABASE_PATH')

n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3

#Reading database
database = pd.read_csv(DATABASE_PATH)

#Preprocessing of data
def get_categories( column ):
    labels = database[column].value_counts().index
    return dict(zip(labels,range(len(labels))))

#Change categorical_columns for values for that category
categorical_columns = set(database.columns)-set(database._get_numeric_data().columns)
clean_database = database.copy()
for col in categorical_columns:
    category = get_categories(col)
    clean_database[col] = clean_database[col].replace(category)

X_train, X_test, y_train, y_test = train_test_split( clean_database.iloc[:,:-1],
                                                    clean_database.iloc[:,-1], test_size=0.3, random_state=42 )

mlflow.set_tracking_uri(MLFLOW_PORT)
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run() as run:
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, "model")
    print(f"Trained and logged model with accuracy: {acc}, run_id: {run.info.run_id}")

