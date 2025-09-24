"""
Daily forecasting script for automated daily scoring - Azure Blob Storage Version.
Production version with Azure Blob Storage, CSV format, and GitHub Actions compatibility.
Enhanced with trading day checks and gap handling.
"""
import os
import sys
import datetime
import smtplib
import json
import logging
import io
import tempfile
from typing import Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from azure.storage.blob import BlobServiceClient
from mkt_lstm import update_fred_data, forecast_n_days
import pandas_market_calendars as mcal
from fredapi import Fred

# Initialize NYSE calendar
nyse = mcal.get_calendar('NYSE')

# Azure Blob Storage configuration
# Allow overriding the container name via environment variable for flexibility in CI
AZURE_CONTAINER_NAME =  'lstm'
HISTORICAL_DATA_BLOB = 'historical_data.csv'
PREDICTED_DATA_BLOB = 'predicted_data.csv'
MODEL_BLOB = 'lstm_model.keras'
SCALER_BLOB = 'scaler.pkl'
METRICS_BLOB = 'metrics.csv'
HEALTH_CHECK_BLOB = 'health_check.json'

# Email configuration
NOTIFICATION_EMAIL = os.getenv('NOTIFICATION_EMAIL', 'jeells@me.com')
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', 587))
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')

# Forecasting configuration
N_DAYS = 30
SEQ_LEN = 10
ROLLING_WINDOW_DAYS = 365

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('daily_forecast.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_blob_service_client():
    """Get Azure Blob Service Client."""
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is required")
    return BlobServiceClient(connection_string)


def ensure_container_exists(container_name: str, logger=None):
    """Ensure the specified container exists; attempt to create it if missing.

    This is best-effort: creation may fail if credentials lack permission, in
    which case we'll surface a warning and allow the calling code to handle the error.
    """
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(container_name)
        try:
            if not container_client.exists():
                # Try to create the container
                try:
                    blob_service_client.create_container(container_name)
                    if logger:
                        logger.info(f"Created missing container: {container_name}")
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to create container {container_name}: {e}")
        except Exception as e:
            # Could not check existence (permission or network issue)
            if logger:
                logger.warning(f"Could not verify container {container_name}: {e}")
    except Exception as e:
        if logger:
            logger.warning(f"Could not connect to blob service to ensure container exists: {e}")

def send_notification(subject, message, is_error=False, logger=None):
    """Send email notification about daily forecast status."""
    if not EMAIL_USER or not EMAIL_PASS:
        msg = f"Email credentials not configured. Would send: {subject}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = NOTIFICATION_EMAIL
        msg['Subject'] = f"Daily Forecast {'ERROR' if is_error else 'SUCCESS'}: {subject}"

        msg.attach(MIMEText(message, 'plain'))

        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        text = msg.as_string()
        server.sendmail(EMAIL_USER, NOTIFICATION_EMAIL, text)
        server.quit()

        if logger:
            logger.info(f"Email notification sent: {subject}")
        else:
            print(f"Email notification sent: {subject}")

    except Exception as e:
        error_msg = f"Failed to send email notification: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)

def upload_to_blob(data, blob_name: str, file_format='csv', logger=None):
    """Upload DataFrame or data to Azure Blob Storage."""
    try:
        # Ensure container exists (best-effort)
        ensure_container_exists(AZURE_CONTAINER_NAME, logger)

        blob_service_client = get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=blob_name
        )

        # Convert to bytes
        if file_format == 'csv':
            buffer = io.StringIO()
            data.to_csv(buffer, index=False)
            content = buffer.getvalue().encode('utf-8')
        elif file_format == 'json':
            content = json.dumps(data).encode('utf-8')
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        blob_client.upload_blob(content, overwrite=True)

        if logger:
            logger.info(f"Successfully uploaded {len(data) if hasattr(data, '__len__') else 'data'} records to {blob_name}")

    except Exception as e:
        error_msg = f"Error uploading to blob {blob_name}: {e}"
        if logger:
            logger.error(error_msg)
        raise

def download_from_blob(blob_name: str, file_format='csv', logger=None):
    """Download data from Azure Blob Storage."""
    try:
        # Ensure container exists (best-effort) to provide clearer errors
        ensure_container_exists(AZURE_CONTAINER_NAME, logger)

        blob_service_client = get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=blob_name
        )

        if not blob_client.exists():
            return None

        content = blob_client.download_blob().readall()

        if file_format == 'csv':
            return pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file_format == 'json':
            return json.loads(content.decode('utf-8'))
        else:
            raise ValueError(f"Unsupported format: {file_format}")

    except Exception as e:
        if logger:
            logger.warning(f"Could not download {blob_name}: {e}")
        return None

def append_to_blob_csv(new_data: pd.DataFrame, blob_name: str, logger=None):
    """Append data to existing CSV file in blob storage."""
    try:
        # Ensure container exists (best-effort)
        ensure_container_exists(AZURE_CONTAINER_NAME, logger)

        # Download existing data
        existing_data = download_from_blob(blob_name, 'csv', logger)

        if existing_data is not None:
            # Remove duplicates based on Date column
            if 'Date' in new_data.columns and 'Date' in existing_data.columns:
                # Convert Date columns to datetime for proper comparison
                existing_data['Date'] = pd.to_datetime(existing_data['Date'])
                new_data['Date'] = pd.to_datetime(new_data['Date'])

                # Keep only new dates not in existing data
                new_dates = new_data['Date'].isin(existing_data['Date'])
                new_data_filtered = new_data[~new_dates]

                if len(new_data_filtered) == 0:
                    if logger:
                        logger.info(f"No new data to append to {blob_name}")
                    return

                combined_data = pd.concat([existing_data, new_data_filtered], ignore_index=True)
            else:
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            combined_data = new_data

        # Upload combined data
        upload_to_blob(combined_data, blob_name, 'csv', logger)

    except Exception as e:
        error_msg = f"Error appending to blob {blob_name}: {e}"
        if logger:
            logger.error(error_msg)
        raise

def get_last_trading_day(reference_date: Optional[datetime.date] = None) -> datetime.date:
    """Get the last trading day before the reference date (default: today)."""
    if reference_date is None:
        reference_date = datetime.date.today()

    # Get trading schedule for the past month to ensure we find the last trading day
    start_date = reference_date - datetime.timedelta(days=30)
    schedule = nyse.schedule(start_date=start_date, end_date=reference_date)

    # Filter to dates before reference_date
    trading_days = [d.date() for d in schedule.index if d.date() < reference_date]

    if not trading_days:
        raise Exception(f"No trading days found before {reference_date}")

    return max(trading_days)

def is_trading_day(date: datetime.date) -> bool:
    """Check if a given date is a trading day."""
    try:
        schedule = nyse.schedule(start_date=date, end_date=date)
        return len(schedule) > 0
    except Exception:
        return False

def get_missing_trading_days(last_data_date: datetime.date,
                           reference_date: Optional[datetime.date] = None) -> list:
    """Get list of trading days that are missing between last_data_date and reference_date."""
    if reference_date is None:
        reference_date = datetime.date.today()

    # Get all trading days between last_data_date and reference_date
    start_date = last_data_date + datetime.timedelta(days=1)

    if start_date >= reference_date:
        return []

    schedule = nyse.schedule(start_date=start_date, end_date=reference_date)
    missing_days = [d.date() for d in schedule.index if d.date() < reference_date]

    return missing_days

def update_fred_data_efficient(start_date: Optional[str] = None) -> pd.DataFrame:
    """More efficient version that only fetches recent data."""
    try:
        fred_api_key = os.getenv('FRED_API_KEY')
        if not fred_api_key:
            raise ValueError("FRED_API_KEY environment variable is required")

        fred = Fred(api_key=fred_api_key)
    except Exception as e:
        raise Exception(f"Error initializing FRED API: {e}")

    # If no start_date provided, default to a reasonable recent period
    if start_date is None:
        start_date = (datetime.date.today() - datetime.timedelta(days=60)).strftime('%Y-%m-%d')

    try:
        sp_500 = fred.get_series('SP500', observation_start=start_date)
        vix = fred.get_series('VIXCLS', observation_start=start_date)
        djia = fred.get_series('DJIA', observation_start=start_date)
        bond = fred.get_series('BAMLCC4A0710YTRIV', observation_start=start_date)
    except Exception as e:
        raise Exception(f"Error fetching data from FRED API: {e}")

    # Process data similar to original function
    df_sp500 = pd.DataFrame(sp_500, columns=['SP500'])
    df_sp500['Date'] = df_sp500.index

    df_vix = pd.DataFrame(vix, columns=['VIXCLS'])
    df_vix['Date'] = df_vix.index

    df_djia = pd.DataFrame(djia, columns=['DJIA'])
    df_djia['Date'] = df_djia.index

    df_bond = pd.DataFrame(bond, columns=['BAMLCC4A0710YTRIV'])
    df_bond['Date'] = df_bond.index
    df_bond = df_bond.rename(columns={'BAMLCC4A0710YTRIV': 'HY_BOND_IDX'})

    df_data = df_sp500.merge(df_vix, on='Date', how='left')
    df_data = df_data.merge(df_djia, on='Date', how='left')
    df_data = df_data.merge(df_bond, on='Date', how='left')
    df_data['Date'] = pd.to_datetime(df_data['Date'])
    df_data.set_index('Date', inplace=True)
    df_data = df_data.dropna()

    # Filter to trading days only
    schedule = nyse.schedule(start_date=df_data.index.min(), end_date=df_data.index.max())
    df_data = df_data[df_data.index.isin(schedule.index)]

    return df_data

def should_run_forecast(logger=None) -> Tuple[bool, str]:
    """Determine if the forecast should run based on trading days and data availability."""
    try:
        today = datetime.date.today()

        # Check if yesterday was a trading day
        yesterday = today - datetime.timedelta(days=1)
        if not is_trading_day(yesterday):
            return False, f"Yesterday ({yesterday}) was not a trading day"

        # Check if we already have data for the last trading day
        existing_data = download_from_blob(HISTORICAL_DATA_BLOB, 'csv', logger)

        if existing_data is not None and len(existing_data) > 0:
            last_data_date = pd.to_datetime(existing_data['Date']).max().date()
            last_trading_day = get_last_trading_day()

            if last_data_date >= last_trading_day:
                return False, f"Data is current. Last data: {last_data_date}, Last trading day: {last_trading_day}"

        # Check if there's actually new data available from FRED
        try:
            # Fetch just the last few days to check for new data
            recent_data = update_fred_data_efficient(
                start_date=(yesterday - datetime.timedelta(days=5)).strftime('%Y-%m-%d')
            )

            if len(recent_data) == 0:
                return False, "No new data available from FRED"

            latest_fred_date = recent_data.index.max().date()

            if existing_data is not None and len(existing_data) > 0:
                if latest_fred_date <= last_data_date:
                    return False, f"No new FRED data. Latest FRED: {latest_fred_date}, Latest local: {last_data_date}"

        except Exception as e:
            if logger:
                logger.warning(f"Could not check FRED data availability: {e}")
            # If we can't check FRED, proceed with caution but allow the run
            return True, f"Could not verify FRED data, proceeding with forecast run"

        return True, f"New data available. Should process trading day {yesterday}"

    except Exception as e:
        if logger:
            logger.error(f"Error in should_run_forecast: {e}")
        return False, f"Error determining if forecast should run: {e}"

def catch_up_missing_data(logger=None) -> bool:
    """Check for and process any missing trading days due to system outages."""
    try:
        existing_data = download_from_blob(HISTORICAL_DATA_BLOB, 'csv', logger)

        if existing_data is None or len(existing_data) == 0:
            if logger:
                logger.info("No existing data found, need to fetch from scratch")
            return True

        last_data_date = pd.to_datetime(existing_data['Date']).max().date()
        missing_days = get_missing_trading_days(last_data_date)

        if not missing_days:
            if logger:
                logger.info("No missing trading days found")
            return False

        if logger:
            logger.info(f"Found {len(missing_days)} missing trading days: {missing_days[:5]}..." if len(missing_days) > 5 else f"Found {len(missing_days)} missing trading days: {missing_days}")

        # Fetch data for missing period
        start_date = missing_days[0].strftime('%Y-%m-%d')
        new_data = update_fred_data_efficient(start_date=start_date)

        if len(new_data) > 0:
            # Convert to DataFrame format for blob storage
            new_data_df = new_data.reset_index().rename(columns={'index': 'Date'})
            append_to_blob_csv(new_data_df, HISTORICAL_DATA_BLOB, logger)

            if logger:
                logger.info(f"Appended {len(new_data_df)} rows of missing data to {HISTORICAL_DATA_BLOB}")
            return True
        else:
            if logger:
                logger.warning("No new data found during catch-up")
            return False

    except Exception as e:
        if logger:
            logger.error(f"Error during catch-up: {e}")
        return False

def load_model_and_scaler_from_blob(logger=None):
    """Load model and scaler from Azure Blob Storage."""
    try:
        blob_service_client = get_blob_service_client()

        # Download model
        model_blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=MODEL_BLOB
        )

        if not model_blob_client.exists():
            raise Exception(f"Model blob {MODEL_BLOB} does not exist")

        model_content = model_blob_client.download_blob().readall()

        # Save to temp file and load
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as temp_file:
            temp_file.write(model_content)
            temp_model_path = temp_file.name

        model = tf.keras.models.load_model(temp_model_path)

        # Clean up temp file
        os.remove(temp_model_path)

        # Download scaler
        scaler_blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=SCALER_BLOB
        )

        if not scaler_blob_client.exists():
            raise Exception(f"Scaler blob {SCALER_BLOB} does not exist")

        scaler_content = scaler_blob_client.download_blob().readall()
        scaler = pickle.loads(scaler_content)

        if logger:
            logger.info("Model and scaler loaded successfully from blob storage")

        return model, scaler

    except Exception as e:
        error_msg = f"Error loading model and scaler from blob: {e}"
        if logger:
            logger.error(error_msg)
        raise Exception(error_msg)

def create_sequences(data, seq_length):
    """Create sequences for LSTM input from time series data."""
    X = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
    return np.array(X)

def generate_daily_prediction(model, scaler, df_data, logger=None):
    """Generate prediction for the next trading day."""
    try:
        if len(df_data) < SEQ_LEN:
            if logger:
                logger.error(f"Not enough data for prediction. Need at least {SEQ_LEN} rows, got {len(df_data)}")
            return None

        # Scale the data
        scaled_data = scaler.transform(df_data.values)

        # Create sequence for prediction
        last_sequence = scaled_data[-SEQ_LEN:].reshape(1, SEQ_LEN, -1)

        # Generate prediction
        prediction_scaled = model.predict(last_sequence, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled)

        # Get the next trading day
        last_date = df_data.index[-1].date()
        next_trading_dates = nyse.valid_days(
            start_date=last_date + pd.Timedelta(days=1),
            end_date=last_date + pd.Timedelta(days=10)
        )

        if len(next_trading_dates) == 0:
            if logger:
                logger.error("Could not find next trading day")
            return None

        next_trading_day = next_trading_dates[0].date()

        # Create prediction DataFrame
        pred_df = pd.DataFrame(prediction, columns=df_data.columns)
        pred_df['Date'] = next_trading_day

        if logger:
            logger.info(f"Generated prediction for {next_trading_day}")

        return pred_df

    except Exception as e:
        error_msg = f"Error generating daily prediction: {e}"
        if logger:
            logger.error(error_msg)
        return None

def compute_rolling_metrics_and_variability(model, scaler, df_historical, df_predicted,
                                          forecast_blobs=None, rolling_days=365, logger=None):
    """Compute rolling metrics on actual vs predicted data."""
    try:
        metrics = {'Date': datetime.date.today()}

        if logger:
            logger.info(f"Computing metrics with {len(df_historical)} historical and {len(df_predicted)} predicted rows")

        if len(df_historical) > 0 and len(df_predicted) > 0:
            # Get the last rolling_days worth of data
            if len(df_historical) < rolling_days:
                actual_data = df_historical.copy()
                pred_data = df_predicted.copy()
            else:
                actual_data = df_historical.iloc[-rolling_days:].copy()
                pred_data = df_predicted.iloc[-rolling_days:].copy()

            # Prepare data for alignment
            if 'Date' not in actual_data.columns and actual_data.index.name != 'Date':
                actual_data = actual_data.reset_index().rename(columns={'index': 'Date'})

            if 'Date' in actual_data.columns:
                actual_data['Date'] = pd.to_datetime(actual_data['Date'])
                actual_data = actual_data.set_index('Date')

            if 'Date' in pred_data.columns:
                pred_data['Date'] = pd.to_datetime(pred_data['Date'])
                pred_data = pred_data.set_index('Date')

            # Find common dates
            common_dates = actual_data.index.intersection(pred_data.index)
            if len(common_dates) > 0:
                actual_aligned = actual_data.loc[common_dates]
                pred_aligned = pred_data.loc[common_dates]

                if logger:
                    logger.info(f"Computing performance metrics on {len(common_dates)} aligned dates")

                # Calculate performance metrics for each asset
                for col in actual_aligned.columns:
                    if col in pred_aligned.columns:
                        actual_vals = actual_aligned[col].values
                        pred_vals = pred_aligned[col].values

                        # Remove any NaN values
                        mask = ~(np.isnan(actual_vals) | np.isnan(pred_vals))
                        if mask.sum() > 0:
                            actual_clean = actual_vals[mask]
                            pred_clean = pred_vals[mask]

                            mse = mean_squared_error(actual_clean, pred_clean)
                            mae = mean_absolute_error(actual_clean, pred_clean)
                            r2 = r2_score(actual_clean, pred_clean)

                            metrics[f'{col}_MSE'] = mse
                            metrics[f'{col}_MAE'] = mae
                            metrics[f'{col}_R2'] = r2

        return pd.DataFrame([metrics])

    except Exception as e:
        error_msg = f"Error computing metrics: {e}"
        if logger:
            logger.error(error_msg)
        # Return basic placeholder on error
        error_metrics = {'Date': datetime.date.today()}
        return pd.DataFrame([error_metrics])

def perform_health_check(logger=None):
    """Perform comprehensive health check and save results to blob storage."""
    health_status = {
        'timestamp': datetime.datetime.now().isoformat(),
        'status': 'healthy',
        'checks': {}
    }

    try:
        # Check Azure Blob Storage connection
        try:
            blob_service_client = get_blob_service_client()
            container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
            container_client.get_container_properties()
            health_status['checks']['azure_blob_accessible'] = True
        except Exception as e:
            health_status['checks']['azure_blob_accessible'] = False
            health_status['checks']['azure_blob_error'] = str(e)
            health_status['status'] = 'unhealthy'

        # Check critical blobs exist
        critical_blobs = {
            'model': MODEL_BLOB,
            'scaler': SCALER_BLOB
        }

        try:
            blob_service_client = get_blob_service_client()
            for name, blob_name in critical_blobs.items():
                try:
                    blob_client = blob_service_client.get_blob_client(
                        container=AZURE_CONTAINER_NAME,
                        blob=blob_name
                    )
                    exists = blob_client.exists()
                    health_status['checks'][f'{name}_exists'] = exists
                    if not exists:
                        health_status['status'] = 'warning'
                except Exception as e:
                    health_status['checks'][f'{name}_error'] = str(e)
                    health_status['status'] = 'warning'
        except Exception as e:
            health_status['checks']['blob_access_error'] = str(e)
            health_status['status'] = 'unhealthy'

        # Check data freshness
        historical_data = download_from_blob(HISTORICAL_DATA_BLOB, 'csv', logger)
        if historical_data is not None and len(historical_data) > 0:
            try:
                last_data_date = pd.to_datetime(historical_data['Date'].max())
                days_since_update = (datetime.datetime.now() - last_data_date).days
                health_status['checks']['data_freshness_days'] = days_since_update
                health_status['checks']['data_is_fresh'] = days_since_update <= 5

                if days_since_update > 5:
                    health_status['status'] = 'warning'
            except Exception as e:
                health_status['checks']['data_freshness_error'] = str(e)
                health_status['status'] = 'warning'

        # Check FRED API key
        api_key = os.getenv('FRED_API_KEY')
        health_status['checks']['fred_api_key_configured'] = api_key is not None and len(api_key) > 0

        if not health_status['checks'].get('fred_api_key_configured', False):
            health_status['status'] = 'unhealthy'

        # Save health check results to blob storage
        upload_to_blob(health_status, HEALTH_CHECK_BLOB, 'json', logger)

        if logger:
            logger.info(f"Health check completed: {health_status['status']}")

        return health_status

    except Exception as e:
        health_status['status'] = 'error'
        health_status['error'] = str(e)

        if logger:
            logger.error(f"Health check failed: {e}")

        return health_status

def main():
    """Main daily forecast execution with enhanced robustness."""
    # Setup logging
    logger = setup_logging()
    run_date = datetime.date.today().strftime('%Y-%m-%d')

    logger.info(f"=" * 80)
    logger.info(f"DAILY FORECAST STARTED - {run_date}")
    logger.info(f"Using Azure Blob Storage container: {AZURE_CONTAINER_NAME}")
    logger.info(f"=" * 80)

    try:
        # Perform initial health check
        health_status = perform_health_check(logger)
        if health_status['status'] == 'unhealthy':
            raise Exception(f"Health check failed: {health_status}")

        # Check if forecast should run
        should_run, reason = should_run_forecast(logger)
        logger.info(f"Should run forecast: {should_run} - {reason}")

        if not should_run:
            # Still perform a final health check and exit gracefully
            final_health = perform_health_check(logger)

            summary = f"""
Daily Forecast Skipped - {run_date}

Reason: {reason}

System Status: {final_health['status']}
            """

            send_notification(f"Daily Forecast Skipped - {run_date}", summary.strip(), logger=logger)
            logger.info("Daily forecast run skipped")
            logger.info(f"=" * 80)
            return

        # Attempt to catch up any missing data first
        catch_up_performed = catch_up_missing_data(logger)

        # Load the most current data (after potential catch-up)
        current_data = download_from_blob(HISTORICAL_DATA_BLOB, 'csv', logger)

        if current_data is not None and len(current_data) > 0:
            # Convert to the expected format for processing
            current_data['Date'] = pd.to_datetime(current_data['Date'])
            df_data = current_data.set_index('Date').sort_index()

            logger.info(f"Loaded current data through {df_data.index[-1]} ({len(df_data)} total rows)")
        else:
            # If no existing data, fetch from scratch
            logger.info("No existing data found, fetching from start date...")
            df_data = update_fred_data(start_day='2020-01-16')
            logger.info(f"Fetched data through {df_data.index[-1]} ({len(df_data)} total rows)")

            # Upload raw data to blob storage as CSV
            df_raw = df_data.reset_index().rename(columns={'index': 'Date'})
            upload_to_blob(df_raw, HISTORICAL_DATA_BLOB, 'csv', logger)

        # Load model and scaler from blob storage
        logger.info("Loading model and scaler from blob storage...")
        model, scaler = load_model_and_scaler_from_blob(logger)

        # Generate and append daily prediction
        logger.info("Generating daily prediction...")
        daily_pred = generate_daily_prediction(model, scaler, df_data, logger)
        if daily_pred is not None:
            append_to_blob_csv(daily_pred, PREDICTED_DATA_BLOB, logger)
        else:
            logger.warning("Could not generate daily prediction")

        # Generate future forecasts
        logger.info("Generating 30-day forecast...")
        last_date = df_data.index[-1]
        future_dates = nyse.valid_days(
            start_date=last_date + pd.Timedelta(days=1),
            end_date=last_date + pd.Timedelta(days=60)
        )[:N_DAYS]

        forecasts = forecast_n_days(model, scaler, df_data, n_days=N_DAYS)
        df_forecast = pd.DataFrame(forecasts, columns=df_data.columns, index=future_dates)
        df_forecast = df_forecast.reset_index().rename(columns={'index': 'Date'})

        # Save forecast to blob with date in filename
        forecast_blob = f'forecasts/forecasted_{run_date}.csv'
        upload_to_blob(df_forecast, forecast_blob, 'csv', logger)
        logger.info(f"Forecast saved to blob: {forecast_blob}")

        # Compute rolling metrics
        logger.info("Computing performance metrics...")

        # Load existing predicted data for metrics calculation
        df_predicted_historical = download_from_blob(PREDICTED_DATA_BLOB, 'csv', logger)

        if df_predicted_historical is not None and len(df_predicted_historical) > 0:
            metrics_df = compute_rolling_metrics_and_variability(
                model, scaler, df_data.reset_index(), df_predicted_historical,
                None, logger=logger
            )
            append_to_blob_csv(metrics_df, METRICS_BLOB, logger)
        else:
            logger.warning("No predicted data available for metrics calculation")

        # Final health check
        final_health = perform_health_check(logger)

        # Send success notification
        summary = f"""
Daily Forecast Completed Successfully - {run_date}

Summary:
- Data fetched through: {df_data.index[-1]}
- Historical data rows: {len(df_data)}
- Daily prediction: {'Generated' if daily_pred is not None else 'Failed'}
- 30-day forecast: Generated ({len(df_forecast)} days)
- Metrics: {'Updated' if df_predicted_historical is not None and len(df_predicted_historical) > 0 else 'Skipped (no prediction history)'}
- Catch-up performed: {'Yes' if catch_up_performed else 'No'}
- Health Status: {final_health['status']}

Blobs Updated:
- {HISTORICAL_DATA_BLOB}
- {PREDICTED_DATA_BLOB}
- {forecast_blob}
- {METRICS_BLOB}

Storage: Azure Blob Storage Container '{AZURE_CONTAINER_NAME}'
        """

        send_notification(f"Daily Forecast Success - {run_date}", summary.strip(), logger=logger)
        logger.info("Daily forecast completed successfully")
        logger.info(f"=" * 80)

    except Exception as e:
        error_message = f"""
Daily Forecast Failed - {run_date}

Error Details:
{str(e)}

Please check GitHub Actions logs for detailed error information.
        """

        logger.error(f"Daily forecast failed: {e}", exc_info=True)
        send_notification(f"Daily Forecast Failed - {run_date}", error_message.strip(),
                         is_error=True, logger=logger)
        logger.info(f"=" * 80)
        raise

if __name__ == '__main__':
    main()
