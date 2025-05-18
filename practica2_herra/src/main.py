import subprocess
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify, render_template

load_dotenv()

FLASK_PORT=os.getenv('FLASK_PORT')
EXPERIMENT_NAME=os.getenv('EXPERIMENT_NAME')
MLFLOW_PORT=os.getenv('MLFLOW_PORT')
DATABASE_PATH=os.getenv('DATABASE_PATH')

# Abrir el servidos de mlflow
mlflow.set_tracking_uri(MLFLOW_PORT)
experiment_name = EXPERIMENT_NAME

def get_best_model():
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        return None
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    best_run_id = runs.iloc[0]["run_id"]
    return mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

def get_newest():
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1
    )
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    best_run_id = runs.iloc[0]["run_id"]
    newest = runs.iloc[0]["run_id"]
    return [best_run_id,newest]
# Load the best model on startup so predictions comes from there
model = get_best_model()
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.get_json()
    print(input_data)
    features = pd.DataFrame([input_data])
    if not model:
        return jsonify({"prediction": "No hay modelos cargados, entrenar primero"})
    prediction = model.predict(features)
    return jsonify({"prediction": "Si" if int(prediction[0]) else "No"})

@app.route("/train", methods=["POST"])
def train():
    try:
        params = request.get_json()
        n_estimators = params.get("n_estimators", 100)
        max_depth = params.get("max_depth", 3)


        # Trigger training script
        subprocess.run(["python3", "train.py", str(n_estimators), str(max_depth)], check=True)
        # Reload best model
        newest = get_newest()
        message = "Entrenamiento terminado, no se mejoró la precisición, utilizando entramiento previo",
        if newest[0] != newest[1]:
            message = "Entrenamiento terminado, utilizandolo para la predicción"

        return jsonify({
            "message": message,
            "n_estimators": n_estimators,
            "max_depth": max_depth
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/test", methods=["POST"])
def test():
    try:
        database = pd.read_csv(DATABASE_PATH)
        testing = database.sample(1)
        global model
        model = get_best_model()
        if not model:
            return jsonify({"prediction": "No hay modelos cargados, entrenar primero"})

        def get_categories( column ):
            labels = database[column].value_counts().index
            return dict(zip(labels,range(len(labels))))

        categorical_columns = set(database.columns)-set(database._get_numeric_data().columns)
        for col in categorical_columns:
            category = get_categories(col)
            testing[col] = testing[col].replace(category)

        print(testing)

        pred = model.predict(testing.iloc[:,:-1])

        return jsonify({
            "data": str(dict( testing.iloc[:,:-1] )),
            "real": "Si" if testing["HeartDisease"].iloc[0] else "No",
            "prediction": "Si" if pred else "No"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=int(FLASK_PORT))
