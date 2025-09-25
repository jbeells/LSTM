"""
Revised daily_forecast_blob.py

Implements the following logical fixes:
- Uploads three datasets: historical_data.csv, predicted_data.csv (historical fit/predict), forecasted_data.csv (30-day future forecast)
- Produces rolling predictions (one-step-ahead) for the last 365 days to feed metrics, in predicted_data.csv
- Produces 30-day-ahead actual forecasts in forecasted_data.csv
"""

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging

from mkt_lstm import update_fred_data, forecast_n_days, ROLLING_WINDOW_DAYS, SEQ_LEN

from azure.storage.blob import BlobServiceClient
import json
import warnings

warnings.filterwarnings('ignore')

AZURE_CONNECTION_STRING = os.getenv('AZURE_CONNECTION_STRING')
CONTAINER_NAME = os.getenv('AZURE_BLOB_CONTAINER')
HISTORICAL_DATA_BLOB = 'historical_data.csv'
PREDICTED_DATA_BLOB = 'predicted_data.csv'
FORECASTED_DATA_BLOB = 'forecasted_data.csv'
SCALER_BLOB = 'scaler.save'
MODEL_BLOB = 'model.save'
HEALTH_CHECK_BLOB = 'health_check.json'
METRICS_BLOB = 'metrics.csv'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# ========================
# Azure blob helpers
# ========================

def load_from_blob(blob_name, as_type="csv"):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        data = blob_client.download_blob().readall()
        if as_type == "csv":
            from io import BytesIO
            return pd.read_csv(BytesIO(data))
        elif as_type == "joblib":
            from io import BytesIO
            return joblib.load(BytesIO(data))
        elif as_type == "json":
            import io
            return json.load(io.BytesIO(data))
    except Exception as e:
        logger.warning(f"Download failed or not found for {blob_name}: {e}")
        return None

def upload_to_blob(data, blob_name, as_type="csv", logger=None):
    if AZURE_CONNECTION_STRING is None or CONTAINER_NAME is None:
        raise ValueError("Azure credentials missing!")
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
    try:
        if as_type == "csv":
            from io import StringIO
            buffer = StringIO()
            data.to_csv(buffer, index=False)
            blob_client.upload_blob(buffer.getvalue(), overwrite=True)
        elif as_type == "joblib":
            from io import BytesIO
            buffer = BytesIO()
            joblib.dump(data, buffer)
            buffer.seek(0)
            blob_client.upload_blob(buffer.read(), overwrite=True)
        elif as_type == "json":
            blob_client.upload_blob(json.dumps(data), overwrite=True)
        if logger: logger.info(f"Uploaded {blob_name} to Azure blob storage.")
    except Exception as e:
        if logger: logger.error(f"Failed to upload {blob_name}: {e}")

# ========================
# Rolling predictions helper
# ========================

def rolling_predictions(model, scaler, df_data: pd.DataFrame, window: int, seq_len: int):
    """
    For each of the last `window` days, predict SP500 using the prev seq_len days.
    Returns DataFrame with Date, Actual SP500, Predicted SP500, and optionally other cols.
    Assumes df_data index is datetime and numeric cols are ['SP500','VIXCLS','DJIA','HY_BOND_IDX']
    """
    numeric_columns = ['SP500', 'VIXCLS', 'DJIA', 'HY_BOND_IDX']
    preds = []
    data = df_data[numeric_columns]
    dates = df_data.index if isinstance(df_data.index, pd.DatetimeIndex) else df_data['Date']
    for i in range(len(df_data) - window, len(df_data)):
        if i - seq_len < 0:
            continue
        X_seq = scaler.transform(data.iloc[i - seq_len:i].values).reshape(1, seq_len, len(numeric_columns))
        pred_scaled = model.predict(X_seq, verbose=0)
        try:
            pred = scaler.inverse_transform(pred_scaled)[0]
            preds.append({
                'Date': dates[i],
                'SP500_Actual': data.iloc[i]['SP500'],
                'SP500_Predicted': pred[0],
                'VIXCLS_Actual': data.iloc[i]['VIXCLS'],
                'VIXCLS_Predicted': pred[1],
                'DJIA_Actual': data.iloc[i]['DJIA'],
                'DJIA_Predicted': pred[2],
                'HY_BOND_IDX_Actual': data.iloc[i]['HY_BOND_IDX'],
                'HY_BOND_IDX_Predicted': pred[3],
            })
        except Exception as e:
            if logger: logger.error(f"Rolling prediction failed on {dates[i]}: {e}")
            continue
    return pd.DataFrame(preds)

# ========================
# Metrics calculation (unchanged)
# ========================

def compute_metrics(pred_df: pd.DataFrame):
    # pred_df must have: ['Date', 'SP500_Actual', 'SP500_Predicted']
    mae = np.mean(np.abs(pred_df['SP500_Actual'] - pred_df['SP500_Predicted']))
    rmse = np.sqrt(np.mean((pred_df['SP500_Actual'] - pred_df['SP500_Predicted']) ** 2))
    return {
        'mae': round(mae, 3),
        'rmse': round(rmse, 3),
    }

# ========================
# Main workflow
# ========================

def main():
    logger.info("Starting daily forecast process.")

    # --- 1. Load model & scaler
    logger.info("Downloading model and scaler from blob storage...")
    scaler = load_from_blob(SCALER_BLOB, as_type="joblib")
    model = load_from_blob(MODEL_BLOB, as_type="joblib")

    # --- 2. Update FRED data
    logger.info("Updating FRED economic data...")
    updated_data = update_fred_data()
    updated_data = updated_data.reset_index() if 'Date' not in updated_data.columns else updated_data
    updated_data['Date'] = pd.to_datetime(updated_data['Date'])
    updated_data.sort_values('Date', inplace=True)
    updated_data = updated_data.drop_duplicates(subset='Date').set_index('Date')
    
    # --- 3. Upload historical_data.csv
    upload_to_blob(updated_data.reset_index(), HISTORICAL_DATA_BLOB, 'csv', logger)
    logger.info("Uploaded updated historical FRED data.")

    # --- 4. Create and upload predicted_data.csv (historical rolling predictions)
    logger.info("Generating rolling window predictions for model backtest (predicted_data.csv)...")
    rolling_pred_df = rolling_predictions(
        model=model,
        scaler=scaler,
        df_data=updated_data,
        window=ROLLING_WINDOW_DAYS,
        seq_len=SEQ_LEN
    )
    # Just SP500 by default for predictions/metrics
    rolling_pred_df_upload = rolling_pred_df[['Date', 'SP500_Actual', 'SP500_Predicted']]
    upload_to_blob(rolling_pred_df_upload, PREDICTED_DATA_BLOB, 'csv', logger)

    # --- 5. Calculate and upload metrics
    logger.info("Calculating model metrics...")
    metrics = compute_metrics(rolling_pred_df_upload)
    upload_to_blob(pd.DataFrame([metrics]), METRICS_BLOB, 'csv', logger)

    # --- 6. Generate and upload forecasted_data.csv (future 30-day forecast)
    logger.info("Generating 30-day ahead forecast...")
    forecast_df = forecast_n_days(model, scaler, updated_data, steps=30, seq_len=SEQ_LEN)
    if not pd.api.types.is_datetime64_any_dtype(forecast_df['Date']):
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    forecast_df = forecast_df.sort_values('Date')
    upload_to_blob(forecast_df, FORECASTED_DATA_BLOB, 'csv', logger)

    logger.info("All files uploaded successfully. Process complete.")

if __name__ == "__main__":
    main()
