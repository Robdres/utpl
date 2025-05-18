import mlflow
from flask import Flask

mlflow.set_tracking_uri(uri="http://local:9090")

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

if __name__ == '__main__':
    app.run(host="localhost", port=8080)
