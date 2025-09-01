import pandas as pd
from flask import Flask, request, jsonify
import joblib
import os
from google.cloud import storage

app = Flask(__name__)
model = None

def load_model():
    model = joblib.load("model.joblib")
    return model

def load_model_cloud():
    storage_client = storage.Client()
    bucket_name = "sid-kubeflow-v1"
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob("bikeshare-model/artifact/model.joblib")
    blob.download_to_filename("model.joblib")
    model = joblib.load("model.joblib")
    return model

@app.route('/predict', methods=['POST'])
def predict():
    # Uncomment this while running from local system 
    # model = load_model()
    
    # Uncomment this while running from cloud
    model = load_model_cloud()
    try : 
        input_json = request.get_json()
        input_df = pd.DataFrame(input_json, index=[0])
        y_predictions = model.predict(input_df)
        response = {'predictions': y_predictions.tolist()}
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5052)))
