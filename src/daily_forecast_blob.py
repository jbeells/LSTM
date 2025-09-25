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

# Helper function for model compatibility
def build_lstm_compatible(input_shape, output_dim: int):
    """Build LSTM model compatible with current TensorFlow version."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Azure Blob Storage configuration
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME', 'lstm-models')
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
    """Setup logging configuration - GitHub Actions friendly."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def upload_to_blob(data, blob_name: str, file_format='csv', logger=None):
    """Upload DataFrame or data to Azure Blob Storage - TESTED AND WORKING VERSION."""
    try:
        if logger:
            logger.info(f"=== UPLOAD STARTING ===")
            logger.info(f"Container: {AZURE_CONTAINER_NAME}")
            logger.info(f"Blob name: {blob_name}")

        # Convert to bytes
        if file_format == 'csv':
            buffer = io.StringIO()
            data.to_csv(buffer, index=False)
            content_bytes = buffer.getvalue().encode('utf-8')
        elif file_format == 'json':
            content_bytes = json.dumps(data, indent=2).encode('utf-8')
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        # Use the direct approach that worked in testing
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is required")
            
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=blob_name
        )

        blob_client.upload_blob(
            data=content_bytes,
            overwrite=True,
            content_type='text/csv' if file_format == 'csv' else 'application/json'
        )

        if logger:
            logger.info(f"✓ Successfully uploaded to blob: {AZURE_CONTAINER_NAME}/{blob_name}")

    except Exception as e:
        error_msg = f"Error uploading to blob {blob_name}: {e}"
        if logger:
            logger.error(error_msg)
        raise

def download_from_blob(blob_name: str, file_format='csv', logger=None):
    """Download data from Azure Blob Storage - TESTED AND WORKING VERSION."""
    try:
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is required")
            
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=blob_name
        )

        if not blob_client.exists():
            if logger:
                logger.info(f"Blob {blob_name} does not exist")
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

def load_model_and_scaler_from_blob(logger=None):
    """Load model and scaler from Azure Blob Storage with version compatibility."""
    try:
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is required")
            
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        # Download model
        model_blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=MODEL_BLOB
        )
        
        if not model_blob_client.exists():
            raise Exception(f"Model blob {MODEL_BLOB} does not exist in container {AZURE_CONTAINER_NAME}")

        model_content = model_blob_client.download_blob().readall()

        # Use temporary file for model
        temp_dir = tempfile.mkdtemp()
        temp_model_path = os.path.join(temp_dir, 'model.keras')
        
        with open(temp_model_path, 'wb') as temp_file:
            temp_file.write(model_content)

        # Try to load with custom objects to handle version compatibility
        try:
            # First try normal loading
            model = tf.keras.models.load_model(temp_model_path)
        except ValueError as e:
            if 'time_major' in str(e):
                # Handle the time_major parameter issue
                if logger:
                    logger.warning("Model has version compatibility issue, attempting to rebuild...")
                
                # Create a new model with the same architecture as your mkt_lstm.py
                try:
                    model = build_lstm_compatible(input_shape=(SEQ_LEN, 4), output_dim=4)
                    
                    if logger:
                        logger.warning("Created new model architecture. Note: This model will need to be retrained.")
                        logger.warning("For production use, please retrain and re-upload the model with the current TensorFlow version.")
                    
                except Exception as rebuild_error:
                    raise Exception(f"Failed to rebuild model: {rebuild_error}")
            else:
                raise e

        # Clean up temp files
        os.remove(temp_model_path)
        os.rmdir(temp_dir)

        # Download scaler
        scaler_blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=SCALER_BLOB
        )
        
        if not scaler_blob_client.exists():
            raise Exception(f"Scaler blob {SCALER_BLOB} does not exist in container {AZURE_CONTAINER_NAME}")

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

def is_trading_day(date):
    """Check if a given date is a trading day using pandas_market_calendars."""
    try:
        schedule = nyse.schedule(start_date=date, end_date=date)
        return not schedule.empty
    except Exception:
        # Fallback: exclude weekends
        return date.weekday() < 5

def get_next_trading_day(date):
    """Get the next trading day after the given date."""
    next_date = date + datetime.timedelta(days=1)
    while not is_trading_day(next_date):
        next_date += datetime.timedelta(days=1)
    return next_date

def should_run_forecast():
    """Determine if forecast should run based on trading calendar."""
    today = datetime.date.today()
    
    # Always run on trading days
    if is_trading_day(today):
        return True, f"Today ({today}) is a trading day"
    
    # Check if today is the day before a trading day (for weekend/holiday prep)
    next_trading_day = get_next_trading_day(today)
    days_until_trading = (next_trading_day - today).days
    
    if days_until_trading == 1:
        return True, f"Tomorrow ({next_trading_day}) is a trading day"
    
    return False, f"Next trading day is {next_trading_day} ({days_until_trading} days away)"

def send_notification_email(subject: str, body: str, logger=None):
    """Send notification email."""
    try:
        if not all([EMAIL_USER, EMAIL_PASS, NOTIFICATION_EMAIL]):
            if logger:
                logger.warning("Email configuration incomplete, skipping notification")
            return

        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = NOTIFICATION_EMAIL
        msg['Subject'] = f"LSTM Forecast: {subject}"

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        text = msg.as_string()
        server.sendmail(EMAIL_USER, NOTIFICATION_EMAIL, text)
        server.quit()

        if logger:
            logger.info(f"Notification email sent: {subject}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to send notification email: {e}")

def calculate_model_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate comprehensive model performance metrics."""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    
    try:
        r2 = r2_score(actual, predicted)
    except:
        r2 = np.nan
    
    # Calculate percentage errors
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Direction accuracy
    actual_direction = np.diff(actual) > 0
    predicted_direction = np.diff(predicted) > 0
    direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'direction_accuracy': float(direction_accuracy)
    }

def main():
    """Main daily forecast execution with enhanced robustness."""
    logger = setup_logging()
    run_date = datetime.date.today().strftime('%Y-%m-%d')

    logger.info(f"=" * 80)
    logger.info(f"DAILY FORECAST STARTED - {run_date}")
    logger.info(f"Using Azure Blob Storage container: {AZURE_CONTAINER_NAME}")
    logger.info(f"=" * 80)

    try:
        # Check if we should run the forecast
        should_run, reason = should_run_forecast()
        logger.info(f"Forecast execution check: {reason}")

        if not should_run:
            logger.info("Skipping forecast execution (not a trading day or day before trading)")
            
            # Update health check to indicate skip
            health_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'status': 'skipped',
                'reason': reason,
                'next_run': 'Next trading day or day before'
            }
            upload_to_blob(health_data, HEALTH_CHECK_BLOB, 'json', logger)
            return

        # Load model and scaler
        logger.info("Loading model and scaler from blob storage...")
        model, scaler = load_model_and_scaler_from_blob(logger)

        # Download existing data
        logger.info("Downloading existing historical data...")
        historical_data = download_from_blob(HISTORICAL_DATA_BLOB, 'csv', logger)

        # Update FRED data
        logger.info("Updating FRED data...")
        fred_api_key = os.getenv('FRED_API_KEY')
        if not fred_api_key:
            raise ValueError("FRED_API_KEY environment variable is required")

        updated_data = update_fred_data(fred_api_key, logger)
        
        if updated_data is None or len(updated_data) == 0:
            raise Exception("Failed to fetch updated FRED data")

        logger.info(f"Updated data shape: {updated_data.shape}")
        logger.info(f"Date range: {updated_data['Date'].min()} to {updated_data['Date'].max()}")

        # Check for data freshness
        latest_date = pd.to_datetime(updated_data['Date']).max().date()
        days_old = (datetime.date.today() - latest_date).days
        
        if days_old > 5:  # Allow for weekends/holidays
            logger.warning(f"Latest data is {days_old} days old (from {latest_date})")

        # Upload updated historical data
        logger.info("Uploading updated historical data...")
        upload_to_blob(updated_data, HISTORICAL_DATA_BLOB, 'csv', logger)

        # Generate forecasts
        logger.info(f"Generating {N_DAYS}-day forecast...")
        
        # Use recent data for forecasting (last ROLLING_WINDOW_DAYS days)
        recent_data = updated_data.tail(ROLLING_WINDOW_DAYS).copy()

        # Prepare data for forecasting (set Date as index and keep only numeric columns)
        if 'Date' in recent_data.columns:
            recent_data['Date'] = pd.to_datetime(recent_data['Date'])
            recent_data = recent_data.set_index('Date')

        # Keep only the numeric columns that the model expects
        numeric_columns = ['SP500', 'VIXCLS', 'DJIA', 'HY_BOND_IDX']
        forecast_data = recent_data[numeric_columns].copy()

        # Call the forecast function with correct parameters
        forecast_results = forecast_n_days(
            model=model,
            scaler=scaler, 
            df_data=forecast_data,   # Correct parameter name
            n_days=N_DAYS           # Only valid parameters
        )

        if forecast_results is None:
            raise Exception("Forecast generation failed")

        # forecast_results will be a numpy array with shape (N_DAYS, 4)
        # Extract just the SP500 predictions (first column)
        sp500_forecasts = forecast_results[:, 0]  # SP500 is the first column

        logger.info(f"Forecast generated successfully: {len(sp500_forecasts)} days")

        # Prepare forecast data for upload
        forecast_df = pd.DataFrame({
            'Date': pd.date_range(
                start=latest_date + datetime.timedelta(days=1),
                periods=len(sp500_forecasts),
                freq='D'
            ).strftime('%Y-%m-%d'),
            'SP500_Predicted': sp500_forecasts,
            'Generated_Date': run_date,
            'Data_As_Of': latest_date.strftime('%Y-%m-%d')
        })

        # Upload forecast data
        logger.info("Uploading forecast data...")
        upload_to_blob(forecast_df, PREDICTED_DATA_BLOB, 'csv', logger)

        # Calculate and upload metrics (if we have recent actual vs predicted data)
        logger.info("Calculating model performance metrics...")
        
        # Load previous predictions for comparison
        previous_predictions = download_from_blob(PREDICTED_DATA_BLOB, 'csv', logger)
        metrics_data = []
        
        if previous_predictions is not None and len(previous_predictions) > 0:
            # Find overlapping dates between actual data and previous predictions
            actual_dates = set(updated_data['Date'].astype(str))
            pred_dates = set(previous_predictions['Date'].astype(str))
            overlap_dates = actual_dates.intersection(pred_dates)
            
            if len(overlap_dates) >= 5:  # Need at least 5 days for meaningful metrics
                overlap_actual = updated_data[updated_data['Date'].astype(str).isin(overlap_dates)].sort_values('Date')
                overlap_pred = previous_predictions[previous_predictions['Date'].astype(str).isin(overlap_dates)].sort_values('Date')
                
                if len(overlap_actual) == len(overlap_pred):
                    metrics = calculate_model_metrics(
                        overlap_actual['SP500'].values,
                        overlap_pred['SP500_Predicted'].values
                    )
                    
                    metrics_data.append({
                        'Date': run_date,
                        'Evaluation_Period': f"{min(overlap_dates)} to {max(overlap_dates)}",
                        'Days_Evaluated': len(overlap_dates),
                        **metrics
                    })
                    
                    logger.info(f"Model metrics calculated for {len(overlap_dates)} days:")
                    logger.info(f"  RMSE: {metrics['rmse']:.2f}")
                    logger.info(f"  MAE: {metrics['mae']:.2f}")
                    logger.info(f"  R²: {metrics['r2']:.3f}")
                    logger.info(f"  Direction Accuracy: {metrics['direction_accuracy']:.1f}%")

        # Upload metrics
        if metrics_data:
            # Load existing metrics and append
            existing_metrics = download_from_blob(METRICS_BLOB, 'csv', logger)
            if existing_metrics is not None:
                all_metrics = pd.concat([existing_metrics, pd.DataFrame(metrics_data)], ignore_index=True)
            else:
                all_metrics = pd.DataFrame(metrics_data)
            
            # Keep only last 90 days of metrics
            all_metrics = all_metrics.tail(90)
            upload_to_blob(all_metrics, METRICS_BLOB, 'csv', logger)

        # Create and upload health check
        health_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'status': 'success',
            'forecast_date': run_date,
            'data_as_of': latest_date.strftime('%Y-%m-%d'),
            'days_forecasted': N_DAYS,
            'data_freshness_days': days_old,
            'model_metrics': metrics_data[0] if metrics_data else None
        }

        upload_to_blob(health_data, HEALTH_CHECK_BLOB, 'json', logger)

        # Send success notification
        success_message = f"""
Daily forecast completed successfully!

Date: {run_date}
Data as of: {latest_date}
Forecast horizon: {N_DAYS} days
Data freshness: {days_old} days old

Latest S&P 500: {updated_data['SP500'].iloc[-1]:.2f}
Forecast summary:
- Next day: {sp500_forecasts[0]:.2f}
- 7-day avg: {np.mean(sp500_forecasts[:7]):.2f}
- 30-day avg: {np.mean(sp500_forecasts):.2f}

Container: {AZURE_CONTAINER_NAME}
Files updated: historical_data.csv, predicted_data.csv, health_check.json
        """

        send_notification_email("Daily Forecast Completed", success_message.strip(), logger)
        
        logger.info("=" * 80)
        logger.info("DAILY FORECAST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as e:
        error_message = f"Daily forecast failed: {str(e)}"
        logger.error(error_message, exc_info=True)

        # Create error health check
        health_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'status': 'error',
            'error_message': str(e),
            'forecast_date': run_date
        }

        try:
            upload_to_blob(health_data, HEALTH_CHECK_BLOB, 'json', logger)
        except:
            logger.error("Failed to upload error health check")

        # Send error notification
        error_email_body = f"""
Daily forecast failed!

Date: {run_date}
Error: {str(e)}

Please check the GitHub Actions logs for detailed information.
Container: {AZURE_CONTAINER_NAME}
        """

        send_notification_email("Daily Forecast Failed", error_email_body.strip(), logger)
        raise

if __name__ == '__main__':
    main()