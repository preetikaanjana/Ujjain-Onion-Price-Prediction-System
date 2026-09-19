"""
Preprocessing Module for Ujjain Onion Price Prediction System.
Cleans raw AGMARKNET market-price records, aggregates multiple records
for the same trading day into a daily price series using the implemented aggregation rule,
validates positive non-zero prices, and ensures strict chronological ordering.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection import load_raw_data, validate_ujjain_data


def clean_mandi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and aggregate the raw Ujjain mandi dataset.
    
    Operations:
    1. Standardize column names to lowercase snake_case.
    2. Parse dates using flexible mixed-format date parser.
    3. Aggregates multiple available market-price records for the same trading day into a clean daily observation using the implemented aggregation rule.
    4. Assert positive non-zero prices.
    5. Sort strictly chronologically.
    """
    df = df.copy()
    
    # Standardize column names
    rename_dict = {}
    for col in df.columns:
        clean_name = col.strip().lower().replace(" ", "_").replace("_x0020_", "_")
        rename_dict[col] = clean_name
    df = df.rename(columns=rename_dict)
    
    # Identify date column
    date_col = 'date' if 'date' in df.columns else ('arrival_date' if 'arrival_date' in df.columns else None)
    if not date_col:
        raise KeyError("Could not find a valid date column ('date' or 'arrival_date').")
        
    df['date'] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True)
    
    # Numeric conversion
    price_cols = [c for c in ['modal_price', 'min_price', 'max_price', 'arrivals_tonnes'] if c in df.columns]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.dropna(subset=['modal_price'])
    
    # Assert positive prices
    df = df[df['modal_price'] > 0]
    
    # Daily aggregation if multiple rows per date
    if df['date'].duplicated().any():
        agg_dict = {
            'state': 'first',
            'district': 'first',
            'market': 'first',
            'commodity': 'first',
            'modal_price': 'mean'
        }
        if 'min_price' in df.columns:
            agg_dict['min_price'] = 'min'
        if 'max_price' in df.columns:
            agg_dict['max_price'] = 'max'
        if 'arrivals_tonnes' in df.columns:
            agg_dict['arrivals_tonnes'] = 'sum'
            
        df = df.groupby('date', as_index=False).agg(agg_dict)
        
    df = df.sort_values('date').reset_index(drop=True)
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    return df


def load_and_preprocess(raw_path: Optional[str] = None, save_processed: bool = True) -> pd.DataFrame:
    """
    Load raw data, clean and aggregate it, and optionally save to data/processed/ujjain_onion_clean.csv.
    """
    raw_df = load_raw_data(raw_path)
    clean_df = clean_mandi_data(raw_df)
    
    if save_processed:
        os.makedirs(os.path.join(PROJECT_ROOT, 'data', 'processed'), exist_ok=True)
        out_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ujjain_onion_clean.csv')
        export_df = clean_df.copy()
        export_df['date'] = export_df['date_str']
        cols_to_save = [c for c in ['date', 'state', 'district', 'market', 'commodity', 'arrivals_tonnes', 'min_price', 'max_price', 'modal_price', 'year', 'month'] if c in export_df.columns]
        export_df[cols_to_save].to_csv(out_path, index=False)
        print(f"Cleaned dataset saved to {out_path} ({len(clean_df)} daily records).")
        
    return clean_df


if __name__ == '__main__':
    print("Testing preprocessing module on 2024-2026 data...")
    cleaned = load_and_preprocess()
    print("Preprocessing completed successfully.")
    print("Summary:")
    print(f"  Shape: {cleaned.shape}")
    print(f"  Date range: {cleaned['date'].min().date()} to {cleaned['date'].max().date()}")
    print(f"  Modal price range: Rs. {cleaned['modal_price'].min():.2f} - Rs. {cleaned['modal_price'].max():.2f}")
