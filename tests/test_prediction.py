import pytest
from src.predict import predict_future, load_model_bundle


def test_model_bundle_loads():
    bundle = load_model_bundle()
    assert 'model' in bundle
    assert 'test_metrics' in bundle
    assert 'MAE' in bundle['test_metrics']
    assert bundle['test_metrics']['MAE'] > 0
    assert 'latest_trained_date' in bundle
    assert bundle['latest_trained_date'].startswith('2026')


@pytest.mark.parametrize("horizon", [1, 7, 14])
def test_predict_future_horizons_2026(horizon):
    res = predict_future(horizon_days=horizon)
    assert res['horizon_days'] == horizon
    assert res['predicted_price'] > 0, "Predicted price must be positive"
    lower, upper = res['estimated_range']
    assert lower <= upper, "Lower bound must be <= upper bound"
    assert len(res['daily_forecasts']) == horizon
    assert res['target_date'].startswith('2026')
    assert res['latest_actual_date'].startswith('2026')


def test_invalid_horizon_raises_error():
    with pytest.raises(ValueError, match="Unsupported horizon"):
        predict_future(horizon_days=25)
