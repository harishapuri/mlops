import pytest


# Mock google.cloud.storage and storage.Client for tests
import sys
import types
from unittest.mock import MagicMock

mock_google = types.ModuleType('google')
mock_cloud = types.ModuleType('cloud')
mock_storage = types.ModuleType('storage')

import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class MockBlob:
    def download_to_filename(self, filename):
        # Write a valid dummy model.joblib file with correct feature names
        feature_names = [
            "temp", "humidity", "season_2", "season_3", "season_4",
            "month_2", "month_3", "month_4", "month_5", "month_6", "month_7", "month_8", "month_9", "month_10", "month_11", "month_12",
            "hour_1", "hour_2", "hour_3", "hour_4", "hour_5", "hour_6", "hour_7", "hour_8", "hour_9", "hour_10", "hour_11", "hour_12", "hour_13", "hour_14", "hour_15", "hour_16", "hour_17", "hour_18", "hour_19", "hour_20", "hour_21", "hour_22", "hour_23",
            "holiday_1", "weekday_1", "weekday_2", "weekday_3", "weekday_4", "weekday_5", "weekday_6", "workingday_1", "weather_2", "weather_3", "weather_4"
        ]
        X = np.zeros((1, len(feature_names)))
        y = np.array([2.0])  # Ensure prediction > 1.0 for test
        model = RandomForestRegressor()
        model.fit(X, y)
        import pandas as pd
        model.feature_names_in_ = np.array(feature_names)  # For sklearn >=1.0 compatibility
        joblib.dump(model, filename)

class MockBucket:
    def blob(self, name):
        return MockBlob()

class MockStorageClient:
    def get_bucket(self, name):
        return MockBucket()

mock_storage.Client = MockStorageClient
mock_cloud.storage = mock_storage
mock_google.cloud = mock_cloud
sys.modules['google'] = mock_google
sys.modules['google.cloud'] = mock_cloud
sys.modules['google.cloud.storage'] = mock_storage

import main
main.model = None  # Reset model in case of previous state
main.app.config['TESTING'] = True
app = main.app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict(client):
    input_data = '{"temp":0.24,"humidity":0.81,"season_2":0,"season_3":0,"season_4":0,"month_2":0,"month_3":0,"month_4":0,"month_5":0,"month_6":0,"month_7":0,"month_8":0,"month_9":0,"month_10":0,"month_11":0,"month_12":0,"hour_1":0,"hour_2":0,"hour_3":0,"hour_4":0,"hour_5":0,"hour_6":0,"hour_7":0,"hour_8":0,"hour_9":0,"hour_10":0,"hour_11":0,"hour_12":0,"hour_13":0,"hour_14":0,"hour_15":0,"hour_16":0,"hour_17":0,"hour_18":0,"hour_19":0,"hour_20":0,"hour_21":0,"hour_22":0,"hour_23":0,"holiday_1":0,"weekday_1":0,"weekday_2":0,"weekday_3":0,"weekday_4":0,"weekday_5":0,"weekday_6":1,"workingday_1":0,"weather_2":0,"weather_3":0,"weather_4":0}'
    response = client.post('/predict',data=input_data,content_type='application/json')
    assert response.status_code == 200
    assert any(val>1.0 for val in response.json["predictions"])

def test_predict_failure(client):
    input_data = {
        "destination": ["No Urgent Place"],
        "passanger": ["Kid(s)"],
        "weather": ["Sunny"],
        "temperature": [80],
        "time": ["10AM"],
        "coupon": ["Bar"],
        "expiration": ["1d"],
        "gender": ["Female"],
        "age": ["21"],
        "maritalStatus": ["Unmarried partner"],
        "has_children": [1],
        "education": ["Some college - no degree"],
        "occupation": ["Unemployed"],
        "income": ["$37500 - $49999"],
        "Bar": ["never"]
    }
    response = client.post('/predict', json=input_data)
    print(response.status_code)
    print(response.json)
    assert response.status_code == 400