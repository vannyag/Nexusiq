# NexusIQ
Medallion data pipeline (DuckDB) + LightGBM demand forecaster + XGBoost churn classifier + Streamlit CXO app.

    pip install -r requirements.txt
    python generate_data.py   # messy raw CSV -> data/raw_sales.csv
    python pipeline.py        # Bronze -> Silver (+quarantine) -> Gold
    python train.py           # trains both models, writes churn scores
    streamlit run app.py      # open http://localhost:8501
