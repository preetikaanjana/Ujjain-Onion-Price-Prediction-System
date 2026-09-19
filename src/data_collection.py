"""
Data Collection and Validation Module for Ujjain Onion Price Prediction System.
Connects directly to the official Government of India AGMARKNET 2.0 REST API
and provides strict validation of genuine Ujjain Onion records.
"""

import os
import sys
from pathlib import Path
import urllib.request
import ssl
import json
import pandas as pd
from typing import Optional, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def validate_ujjain_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strictly validate that dataset contains genuine records for:
    State: Madhya Pradesh
    Market: Ujjain APMC
    Commodity: Onion
    
    Raises:
        ValueError: If zero valid Ujjain Onion records are found.
    """
    col_map = {c.lower(): c for c in df.columns}
    
    market_col = col_map.get('market')
    commodity_col = col_map.get('commodity')
    state_col = col_map.get('state')
    
    if not (market_col and commodity_col):
        raise ValueError(f"Missing required columns in dataset. Found: {list(df.columns)}")
    
    mask = (
        df[market_col].astype(str).str.contains('Ujjain', case=False, na=False) &
        df[commodity_col].astype(str).str.contains('Onion', case=False, na=False)
    )
    if state_col:
        mask = mask & df[state_col].astype(str).str.contains('Madhya Pradesh', case=False, na=False)
        
    ujjain_onion = df[mask].copy()
    
    # CRITICAL RULE: Fail loudly if no verified Ujjain Onion records exist
    if len(ujjain_onion) == 0:
        raise ValueError(
            "No verified Ujjain Onion records were found. "
            "Do not continue with synthetic data."
        )
        
    return ujjain_onion


def fetch_agmarknet_api(year: int, month: int) -> pd.DataFrame:
    """
    Fetch real-time official mandi data from AGMARKNET 2.0 REST API.
    Endpoint: /prices-and-arrivals/date-wise/specific-commodity
    State ID 19 = Madhya Pradesh, Commodity ID 23 = Onion.
    """
    ctx = ssl._create_unverified_context()
    url = f"https://api.agmarknet.gov.in/v1/prices-and-arrivals/date-wise/specific-commodity?year={year}&month={month}&includeExcel=false&stateId=19&commodityId=23"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
    
    records = []
    with urllib.request.urlopen(req, context=ctx, timeout=15) as res:
        data = json.loads(res.read().decode())
        markets = data.get('markets', [])
        for mkt in markets:
            m_name = mkt.get('marketName', '')
            if 'ujjain' in m_name.lower():
                for d in mkt.get('dates', []):
                    arr_date = d.get('arrivalDate')
                    for row in d.get('data', []):
                        records.append({
                            'state': 'Madhya Pradesh',
                            'district': 'Ujjain',
                            'market': m_name,
                            'commodity': 'Onion',
                            'variety': row.get('variety', 'Onion'),
                            'arrival_date': arr_date,
                            'arrivals_tonnes': row.get('arrivals'),
                            'min_price': row.get('minimumPrice'),
                            'max_price': row.get('maximumPrice'),
                            'modal_price': row.get('modalPrice'),
                            'year': year,
                            'month': month
                        })
    return pd.DataFrame(records)


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the raw official dataset. Checks local paths in order of preference:
    1. data/processed/ujjain_onion_clean.csv (699 daily aggregated records)
    2. data/raw/ujjain_onion_raw.csv (1086 market price records)
    """
    candidates = [
        filepath,
        os.path.join(PROJECT_ROOT, 'data', 'processed', 'ujjain_onion_clean.csv'),
        os.path.join(PROJECT_ROOT, 'data', 'raw', 'ujjain_onion_raw.csv'),
        os.path.join(PROJECT_ROOT, 'data', 'ujjain_onion_prices.csv')
    ]
    for p in candidates:
        if p and os.path.exists(p):
            df = pd.read_csv(p)
            validated_df = validate_ujjain_data(df)
            return validated_df
            
    raise FileNotFoundError(
        "Could not locate verified Ujjain onion data in candidate paths. "
        "Please ensure data/processed/ujjain_onion_clean.csv exists."
    )


def summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute dataset summary statistics for reporting and validation.
    """
    date_col = 'date' if 'date' in df.columns else ('arrival_date' if 'arrival_date' in df.columns else 'Arrival_Date')
    dates = pd.to_datetime(df[date_col], format='mixed', dayfirst=True)
    modal_col = 'modal_price' if 'modal_price' in df.columns else 'Modal_Price'
    
    # 2026 slice
    mask_2026 = dates.dt.year == 2026
    df_2026 = df[mask_2026]
    
    summary = {
        'num_records': len(df),
        'records_2026': int(mask_2026.sum()),
        'earliest_date': str(dates.min().date()),
        'latest_date': str(dates.max().date()),
        'unique_dates': int(dates.nunique()),
        'min_modal_price': float(df[modal_col].min()),
        'max_modal_price': float(df[modal_col].max()),
        'mean_modal_price': round(float(df[modal_col].mean()), 2),
        'mean_modal_2026': round(float(df_2026[modal_col].mean()), 2) if len(df_2026) > 0 else 0.0,
        'latest_2026_price': float(df[modal_col].iloc[-1])
    }
    return summary


if __name__ == '__main__':
    print("Executing data_collection.py on 2024-2026 dataset...")
    df = load_raw_data()
    summary = summarize_dataset(df)
    print("Verification Passed:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
