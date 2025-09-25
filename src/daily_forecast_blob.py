import os
import logging
import pandas as pd
from datetime import datetime
import json
import smtplib
from email.mime.text import MIMEText

from mkt_lstm import update_fred_data, load_latest_model_and_scaler

# --- CONFIGURATION ---
ROLLING_WINDOW_DAYS = 365            # How many days for historical rolling prediction/metrics
SEQ_LEN = 20                         # Model sequence length
PREDICT_AHEAD_DAYS = 30              # Number of days in the future to forecast

HISTORICAL_DATA_BLOB = 'historical_data.csv'
PREDICTED_DATA_BLOB = 'predicted_data.csv'
FORECASTED_DATA_BLOB = 'forecasted_data.csv'
HEALTH_CHECK_BLOB = 'health_check.json'
MODEL_METRICS_BLOB = 'metrics.csv'

def get_logger():
    logger = logging.getLogger("DailyForecast")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(ch)
    return logger

def send_notification_email(subject, body, recipients):
    smtp_server = os.environ.get("SMTP_SERVER")
    smtp_port = os.environ.get("SMTP_PORT", 587)
    smtp_username = os.environ.get("SMTP_USERNAME")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    sender_email = os.environ.get("SENDER_EMAIL")
    if not (smtp_server and smtp_username and smtp_password and sender_email):
        return False
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipients
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(sender_email, recipients, msg.as_string())
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def upload_to_blob(df, blob_name, filetype, logger):
    """
    Save DataFrame as .csv or .json locally and upload to Azure blob.
    Implement your actual Azure logic here if not already.
    """
    path = blob_name
    if filetype == 'csv':
        df.to_csv(path, index=False)
    elif filetype == 'json':
        df.to_json(path, orient='records', lines=True)
    logger.info(f"Saved and uploaded {blob_name}")

def rolling_predictions(model, scaler, df_data: pd.DataFrame, rolling_window: int, seq_len: int):
    """
    Slide a rolling window over the last `rolling_window` days, and for each, predict using prior `seq_len` obs.
    Returns DataFrame: Date, SP500_Actual, SP500_Predicted (and you can add more fields if needed).
    """
    numeric_columns = ['SP500', 'VIXCLS', 'DJIA', 'HY_BOND_IDX']
    preds = []
    data = df_data[numeric_columns]
    dates = df_data["Date"] if "Date" in df_data.columns else df_data.index

    for i in range(len(df_data) - rolling_window, len(df_data)):
        if i - seq_len < 0:
            continue  # not enough data for endpoint
        X_seq = scaler.transform(data.iloc[i - seq_len : i].values).reshape(1, seq_len, len(numeric_columns))
        pred_scaled = model.predict(X_seq, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0]
        preds.append({
            'Date': dates.iloc[i],
            'SP500_Actual': data.iloc[i]['SP500'],
            'SP500_Predicted': pred[0],
        })
    return pd.DataFrame(preds)

def main():
    logger = get_logger()
    logger.info("Starting daily forecast workflow...")

    logger.info("Step 1: Download and update historical data from FRED")
    updated_data = update_fred_data()
    logger.info(f"Updated FRED data: {len(updated_data)} rows")
    upload_to_blob(updated_data, HISTORICAL_DATA_BLOB, 'csv', logger)

    logger.info("Step 2: Load latest model and scaler")
    model, scaler = load_latest_model_and_scaler()

    logger.info("Step 3: Generate and upload rolling historical predictions (for metrics)")
    pred_df = rolling_predictions(model, scaler, updated_data.reset_index() if not "Date" in updated_data else updated_data, ROLLING_WINDOW_DAYS, SEQ_LEN)
    upload_to_blob(pred_df, PREDICTED_DATA_BLOB, 'csv', logger)

    logger.info("Step 4: Generate and upload 30-day future forecast")
    # Use last SEQ_LEN days for prediction; supply as batch
    last_sequence = updated_data[['SP500', 'VIXCLS', 'DJIA', 'HY_BOND_IDX']].iloc[-SEQ_LEN:]
    fcst_dates = pd.bdate_range(updated_data["Date"].max(), periods=PREDICT_AHEAD_DAYS+1, closed='right')
    predictions = []
    data = updated_data[['SP500', 'VIXCLS', 'DJIA', 'HY_BOND_IDX']].copy()
    for fcst_date in fcst_dates:
        input_seq = scaler.transform(data.iloc[-SEQ_LEN:].values).reshape(1, SEQ_LEN, 4)
        fcst_scaled = model.predict(input_seq, verbose=0)
        fcst_unscaled = scaler.inverse_transform(fcst_scaled)[0]
        predictions.append({
            'Date': fcst_date,
            'SP500_Predicted': fcst_unscaled[0]
        })
        next_row = pd.Series(fcst_unscaled, index=['SP500', 'VIXCLS', 'DJIA', 'HY_BOND_IDX'])
        data = pd.concat([data, next_row.to_frame().T], ignore_index=True)
    forecast_df = pd.DataFrame(predictions)
    upload_to_blob(forecast_df, FORECASTED_DATA_BLOB, 'csv', logger)

    logger.info("All steps completed")

if __name__ == "__main__":
    main()