#!/usr/bin/env bash
set -e
python generate_data.py
python pipeline.py
python train.py
streamlit run app.py
