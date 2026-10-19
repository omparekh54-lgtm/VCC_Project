import asyncio
import csv
import importlib.util
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


backend = load_module('watchdog_app', 'backend/app.py')
training = load_module('watchdog_train', 'ml/train.py')
inference = load_module('watchdog_inference', 'ml/inference.py')


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, 'LOG_PATH', tmp_path / 'logs' / 'metrics.csv')
    monkeypatch.setattr(backend, 'state', backend.ServerState())
    monkeypatch.setattr(backend, 'MODEL_PATH', tmp_path / 'missing.joblib')
    monkeypatch.setattr(backend, 'SAGEMAKER_ENDPOINT_NAME', '')
    monkeypatch.setattr(backend, 'predictor', backend._Predictor())
    with TestClient(backend.app) as c:
        yield c


def test_dashboard_and_business_only_logging(client):
    assert client.get('/health').json() == {'status': 'ok'}
    page = client.get('/')
    assert page.status_code == 200 and 'Load generator' in page.text
    metrics = client.get('/metrics').json()
    assert set(metrics) == set(training.FEATURES)
    prediction = client.get('/prediction').json()
    assert prediction['available'] is False
    assert prediction['would_scale_out'] is None
    with backend.LOG_PATH.open() as f:
        assert len(list(csv.DictReader(f))) == 0
    assert client.get('/api/simulate?intensity=1').status_code == 200
    with backend.LOG_PATH.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert float(rows[0]['response_time_ms']) > 0
    assert float(rows[0]['active_connections']) == 0
    assert client.get('/metrics').json()['active_connections'] == 0
    assert client.get('/prediction').json()['recent_requests'] == 1


def test_prediction_local_and_sagemaker_agree(client, tmp_path, monkeypatch):
    path = tmp_path / 'model.joblib'
    forest = DummyRegressor(strategy='constant', constant=600).fit(np.zeros((2, 4)), [0, 0])
    joblib.dump(forest, path)
    meta = {'features': training.FEATURES, 'horizon_s': 30,
            'predicts': 'change_from_recent_latency'}
    path.with_name('model_meta.json').write_text(json.dumps(meta))
    monkeypatch.setattr(backend, 'MODEL_PATH', path)
    backend.state.active_connections = 1
    prediction = client.get('/prediction').json()
    assert prediction['available'] is True
    assert prediction['predicted_ms'] == 600
    assert prediction['would_scale_out'] is True
    model = inference.model_fn(str(tmp_path))
    values, horizon = inference.predict_fn([prediction['features']], model)
    assert values[0] == prediction['predicted_ms'] and horizon == 30
    output, content_type = inference.output_fn((values, horizon))
    assert json.loads(output)['predicted_response_time_ms'] == [600]
    assert content_type == 'application/json'


def test_legacy_three_feature_model(client, tmp_path, monkeypatch):
    path = tmp_path / 'model.joblib'
    forest = DummyRegressor(strategy='constant', constant=125).fit(np.zeros((2, 3)), [0, 0])
    joblib.dump(forest, path)
    monkeypatch.setattr(backend, 'MODEL_PATH', path)
    result = client.get('/prediction').json()
    assert result['predicted_ms'] == 125 and result['horizon_s'] is None
    model = inference.model_fn(str(tmp_path))
    values, horizon = inference.predict_fn([result['features']], model)
    assert values[0] == 125 and horizon is None


def test_sagemaker_response(client, monkeypatch):
    class Endpoint:
        def invoke_endpoint(self, **kwargs):
            assert kwargs['EndpointName'] == 'test-endpoint'
            assert set(json.loads(kwargs['Body'])) == set(training.FEATURES)
            return {'Body': io.BytesIO(b'{"predicted_response_time_ms":[200],"horizon_s":30}')}
    monkeypatch.setattr(backend, 'SAGEMAKER_ENDPOINT_NAME', 'test-endpoint')
    backend.predictor._sagemaker = Endpoint()
    result = client.get('/prediction').json()
    assert result['source'] == 'SageMaker endpoint'
    assert result['predicted_ms'] == 200


def test_recent_latency_excludes_unfinished_requests():
    data = pd.DataFrame({'timestamp': [0., 5., 12.], 'response_time_ms': [1000., 10000., 1000.]})
    np.testing.assert_allclose(training.recent_latency_at(data, np.array([0., 6., 12., 16.])),
                               [0., 1000., 0., 5500.])


def test_time_split_purges_future_targets():
    data = pd.DataFrame({'timestamp': np.arange(200, dtype=float), 'cpu_usage': 30.,
                         'active_connections': 2., 'lag_ms': 1., 'recent_latency_ms': 100.,
                         'response_time_ms': 100.})
    targets = training.add_targets(data, 30, 10)
    train, test = training.time_split(targets, .2, 30, 10)
    assert (train.timestamp + 40 <= test.timestamp.iloc[0]).all()
    assert len(train) > 50 and len(test) > 10
    np.testing.assert_allclose(targets.future_latency_ms, 100.)


def test_training_data_errors_are_actionable():
    with pytest.raises(ValueError, match='No valid training rows'):
        training.add_targets(pd.DataFrame(), 30, 10)
    with pytest.raises(ValueError, match='Not enough rows'):
        training.time_split(pd.DataFrame(), .2, 30, 10)
    with pytest.raises(ValueError, match='test_fraction'):
        training.time_split(pd.DataFrame(), 1., 30, 10)


def test_idle_forecast_does_not_request_scaling(client, monkeypatch):
    monkeypatch.setattr(backend.predictor, 'predict', lambda features: (600., 30., 'local model'))
    result = client.get('/prediction').json()
    assert result['available'] and result['predicted_ms'] == 600
    assert result['would_scale_out'] is False
    assert result['scaling_suppressed_reason'] == 'no recent business traffic'


def test_lambda_suppresses_idle_forecast(monkeypatch):
    import sys
    from types import SimpleNamespace
    monkeypatch.setenv('SAGEMAKER_ENDPOINT_NAME', 'test')
    monkeypatch.setenv('ASG_NAME', 'test')
    monkeypatch.setitem(sys.modules, 'boto3', SimpleNamespace(client=lambda name: object()))
    guard = load_module('watchdog_guard', 'lambda/lambda_function.py')
    monkeypatch.setattr(guard, 'fetch_metrics', lambda: {'active_connections': 0, 'recent_latency_ms': 0})
    monkeypatch.setattr(guard, 'predict_latency', lambda metrics: 600.)
    monkeypatch.setattr(guard, 'put_metric', lambda *args: None)
    def unexpected_scale():
        raise AssertionError('idle forecast must not scale out')
    monkeypatch.setattr(guard, 'scale_out', unexpected_scale)
    assert guard.run_check() == (600., 'idle: no recent business traffic')
