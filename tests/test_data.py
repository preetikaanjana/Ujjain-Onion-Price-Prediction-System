import pytest
import pandas as pd
from src.data_collection import load_raw_data, validate_ujjain_data, summarize_dataset
from src.preprocessing import clean_mandi_data, load_and_preprocess


def test_ujjain_2026_data_loaded_and_valid():
    df = load_raw_data()
    assert len(df) > 0, "Dataset must not be empty"
    assert (df['commodity'].str.lower() == 'onion').all(), "All records must be Onion"
    assert df['market'].str.contains('Ujjain', case=False).all(), "All records must be Ujjain APMC"
    assert df['state'].str.contains('Madhya Pradesh', case=False).all(), "State must be Madhya Pradesh"
    
    # Check 2026 records presence
    dates = pd.to_datetime(df['date'])
    assert (dates.dt.year == 2026).any(), "Dataset must contain 2026 records"
    assert dates.max().year == 2026, "Latest date must be in 2026"


def test_data_validation_fails_on_fake_or_empty_market():
    fake_df = pd.DataFrame({
        'state': ['Maharashtra'],
        'market': ['Lasalgaon'],
        'commodity': ['Onion'],
        'modal_price': [1500]
    })
    with pytest.raises(ValueError, match="No verified Ujjain Onion records were found"):
        validate_ujjain_data(fake_df)


def test_preprocessing_produces_clean_schema():
    cleaned = load_and_preprocess(save_processed=False)
    assert 'date' in cleaned.columns
    assert 'modal_price' in cleaned.columns
    assert (cleaned['modal_price'] > 0).all(), "All modal prices must be strictly positive"
    dates = pd.to_datetime(cleaned['date'])
    assert dates.is_monotonic_increasing, "Dataset must be strictly chronologically sorted"
