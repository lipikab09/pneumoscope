#!/usr/bin/env bash
set -e
pip install -r requirements.txt
python download_data.py
python -m src.train --model densenet121 --epochs 12
python -m src.evaluate
streamlit run app/dashboard.py
