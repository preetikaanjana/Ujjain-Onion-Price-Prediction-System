"""
Ujjain Onion Price Predictor - Entrypoint
"""
import os
import sys

# Forward directly to app/streamlit_app.py
app_path = os.path.join(os.path.dirname(__file__), 'app', 'streamlit_app.py')
with open(app_path, 'r', encoding='utf-8') as f:
    code = f.read()

exec(compile(code, app_path, 'exec'))
