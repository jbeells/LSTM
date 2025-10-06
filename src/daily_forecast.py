#!/usr/bin/env python3
"""
Daily LSTM Forecast Script
==========================

This script performs daily forecasting of financial market data using a pre-trained LSTM model.
It fetches the latest data from FRED, processes it, generates predictions, and sends notifications.

Features:
- Scans FRED for new daily close data (SP500, VIXCLS, DJIA, BAMLCC4A0710YTRIV)
- Handles weekends, holidays, and system outages gracefully
- Uses pre-trained LSTM model for scoring and forecasting
- Generates comprehensive metrics and outputs
- Sends email notifications on failure
- Comprehensive logging throughout the process

Author: AI Assistant
Date: 2024-09-26
"""

import os
import sys
import pickle
import warnings
import logging
import traceback
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import functions from our main LSTM module
try:
    from mkt_lstm import update_fred_data, forecast_n_days, create_sequences, SEQ_LEN
    logging.info("Successfully imported functions from mkt_lstm.py")
except ImportError as e:
    logging.error(f"Failed to import from mkt_lstm.py: {e}")
    sys.exit(1)

# --- Configuration & Setup ---
load_dotenv()
warnings.filterwarnings('ignore')

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"daily_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_UPLOAD = os.path.join(PROJECT_ROOT, 'data', 'upload')
DATA_OUTPUT = os.path.join(PROJECT_ROOT, 'data', 'output')
MODEL_OUTPUT = os.path.join(PROJECT_ROOT, 'models')
REQUIRED_COLUMNS = ['Date', 'SP500', 'VIXCLS', 'DJIA', 'HY_BOND_IDX']
DAYS_TO_KEEP = 365
FORECAST_DAYS = 30

# Ensure directories exist
os.makedirs(DATA_UPLOAD, exist_ok=True)
os.makedirs(DATA_OUTPUT, exist_ok=True)
os.makedirs(MODEL_OUTPUT, exist_ok=True)

logger.info(f"Logging initialized. Log file: {log_path}")
logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Upload directory: {DATA_UPLOAD}")


def send_failure_notification(error_message: str, detailed_error: str = None) -> None:
    """
    Send email notification when the daily forecast fails.
    
    Args:
        error_message: Brief error description
        detailed_error: Detailed error traceback
    """
    try:
        logger.info("Preparing to send failure notification email")
        
        # Email configuration from environment variables
        email_user = os.getenv('EMAIL_USER')
        email_pass = os.getenv('EMAIL_PASS')
        email_host = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
        email_port = int(os.getenv('EMAIL_PORT', '587'))
        notification_email = os.getenv('NOTIFICATION_EMAIL')
        
        if not all([email_user, email_pass, notification_email]):
            logger.warning("Email configuration incomplete. Cannot send notification.")
            return
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_user
        msg['To'] = notification_email
        msg['Subject'] = f"Daily LSTM Forecast Failed - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        body = f"""
        Daily LSTM Forecast Process Failed
        ==================================
        
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Error: {error_message}
        
        Detailed Error Information:
        {detailed_error if detailed_error else 'No additional details available'}
        
        Log File Location: {log_path}
        
        Please investigate and resolve the issue.
        
        Best regards,
        LSTM Forecasting System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(email_host, email_port)
        server.starttls()
        server.login(email_user, email_pass)
        text = msg.as_string()
        server.sendmail(email_user, notification_email, text)
        server.quit()
        
        logger.info("Failure notification email sent successfully")
        
    except Exception as e:
        logger.error(f"Failed to send notification email: {e}")


def is_market_open_today() -> Tuple[bool, str]:
    """
    Check if the market is open today using NYSE calendar.
    
    Returns:
        Tuple of (is_open, reason_message)
    """
    logger.info("Checking if market is open today")
    
    try:
        nyse = mcal.get_calendar('NYSE')
        today = datetime.now().date()
        
        # Get today's schedule
        schedule = nyse.schedule(start_date=today, end_date=today)
        
        if schedule.empty:
            if today.weekday() >= 5:  # Saturday or Sunday
                reason = f"Market closed: Weekend ({today.strftime('%A')})"
            else:
                reason = f"Market closed: Holiday or special closure"
            logger.info(reason)
            return False, reason
        
        logger.info("Market is open today")
        return True, "Market is open"
        
    except Exception as e:
        error_msg = f"Error checking market status: {e}"
        logger.error(error_msg)
        return False, error_msg


def load_model_and_scaler() -> Tuple[Optional[tf.keras.Model], Optional[Any]]:
    """
    Load the pre-trained LSTM model and scaler from the models directory.
    
    Returns:
        Tuple of (model, scaler) or (None, None) if loading fails
    """
    logger.info("Loading pre-trained model and scaler")
    
    model_path = os.path.join(MODEL_OUTPUT, 'lstm_model.keras')
    scaler_path = os.path.join(MODEL_OUTPUT, 'scaler.pkl')
    
    try:
        # Check if files exist
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None, None
        
        if not os.path.exists(scaler_path):
            logger.error(f"Scaler file not found: {scaler_path}")
            return None, None
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Load scaler
        logger.info(f"Loading scaler from {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded successfully")
        
        return model, scaler
        
    except Exception as e:
        logger.error(f"Error loading model and scaler: {e}")
        return None, None


def get_recent_data(df_data: pd.DataFrame, days: int = DAYS_TO_KEEP) -> pd.DataFrame:
    """
    Get the most recent N days of data from the dataframe.
    
    Args:
        df_data: Full dataframe with historical data
        days: Number of recent days to keep
        
    Returns:
        DataFrame with recent data
    """
    logger.info(f"Extracting most recent {days} days of data")
    
    if len(df_data) <= days:
        logger.info(f"Dataset has {len(df_data)} rows, returning all data")
        return df_data.copy()
    
    recent_data = df_data.tail(days).copy()
    logger.info(f"Extracted {len(recent_data)} rows of recent data")
    
    return recent_data


def generate_predictions(model: tf.keras.Model, scaler: Any, df_actuals_yr: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions for the actual year data using the pre-trained model.
    
    Args:
        model: Pre-trained LSTM model
        scaler: Fitted scaler object
        df_actuals_yr: DataFrame with recent year's actual data
        
    Returns:
        DataFrame with predictions
    """
    logger.info("Generating predictions for recent year data")
    
    try:
        # Prepare data for prediction
        numeric_data = df_actuals_yr.drop('Date', axis=1)
        scaled_data = scaler.transform(numeric_data)
        
        # Create sequences for prediction
        X, _ = create_sequences(scaled_data, SEQ_LEN)
        
        if len(X) == 0:
            logger.error(f"Not enough data for sequences. Need at least {SEQ_LEN + 1} rows")
            return pd.DataFrame()
        
        logger.info(f"Created {len(X)} sequences for prediction")
        
        # Generate predictions
        predictions_scaled = model.predict(X, verbose=0)
        predictions = scaler.inverse_transform(predictions_scaled)
        
        # Create prediction DataFrame
        pred_dates = df_actuals_yr['Date'].iloc[SEQ_LEN:].reset_index(drop=True)
        df_predicted = pd.DataFrame(predictions, columns=numeric_data.columns)
        df_predicted.insert(0, 'Date', pred_dates)
        
        logger.info(f"Generated predictions for {len(df_predicted)} data points")
        return df_predicted
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        return pd.DataFrame()


def calculate_model_metrics(df_actuals: pd.DataFrame, df_predicted: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate model performance metrics comparing actuals vs predictions.
    
    Args:
        df_actuals: DataFrame with actual values
        df_predicted: DataFrame with predicted values
        
    Returns:
        DataFrame with metrics
    """
    logger.info("Calculating model performance metrics")
    
    try:
        # Align the dataframes by date (predictions start from SEQ_LEN+1)
        actuals_aligned = df_actuals.iloc[SEQ_LEN:].reset_index(drop=True)
        
        if len(actuals_aligned) != len(df_predicted):
            logger.warning(f"Length mismatch: actuals={len(actuals_aligned)}, predicted={len(df_predicted)}")
            min_len = min(len(actuals_aligned), len(df_predicted))
            actuals_aligned = actuals_aligned.head(min_len)
            df_predicted_temp = df_predicted.head(min_len)
        else:
            df_predicted_temp = df_predicted
        
        metrics_data = {'Date': datetime.now().strftime('%Y-%m-%d')}
        
        # Calculate metrics for each column
        numeric_cols = [col for col in REQUIRED_COLUMNS if col != 'Date']
        
        for col in numeric_cols:
            if col in actuals_aligned.columns and col in df_predicted_temp.columns:
                actual_values = actuals_aligned[col].values
                predicted_values = df_predicted_temp[col].values
                
                # Calculate metrics
                mse = mean_squared_error(actual_values, predicted_values)
                mae = mean_absolute_error(actual_values, predicted_values)
                r2 = r2_score(actual_values, predicted_values)
                
                metrics_data[f'{col}_MSE'] = mse
                metrics_data[f'{col}_MAE'] = mae
                metrics_data[f'{col}_R2'] = r2
                
                logger.info(f"Metrics for {col}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        
        df_metrics = pd.DataFrame([metrics_data])
        logger.info("Model metrics calculated successfully")
        
        return df_metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return pd.DataFrame()


def generate_forecast(model: tf.keras.Model, scaler: Any, df_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate future forecasts using the forecast_n_days function.
    
    Args:
        model: Pre-trained LSTM model
        scaler: Fitted scaler object
        df_data: Historical data for forecasting
        
    Returns:
        DataFrame with forecasted values
    """
    logger.info(f"Generating {FORECAST_DAYS}-day forecast")
    
    try:
        # Use the imported forecast_n_days function
        forecasts = forecast_n_days(model, scaler, df_data, n_days=FORECAST_DAYS)
        
        # Generate future business dates
        nyse = mcal.get_calendar('NYSE')
        last_date = df_data['Date'].iloc[-1]
        
        # Get next 60 business days to ensure we have enough for 30 forecasts
        future_dates = nyse.valid_days(
            start_date=last_date + pd.Timedelta(days=1), 
            end_date=last_date + pd.Timedelta(days=60)
        )[:FORECAST_DAYS]
        
        # Create forecast DataFrame
        numeric_cols = [col for col in REQUIRED_COLUMNS if col != 'Date']
        df_forecast = pd.DataFrame(forecasts, columns=numeric_cols, index=future_dates)
        df_forecast.reset_index(inplace=True)
        df_forecast.rename(columns={'index': 'Date'}, inplace=True)
        
        logger.info(f"Generated forecast for {len(df_forecast)} future business days")
        return df_forecast
        
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        return pd.DataFrame()


def validate_dataframes(df_actuals: pd.DataFrame, df_predicted: pd.DataFrame, df_forecast: pd.DataFrame) -> bool:
    """
    Validate that all dataframes contain the required columns.
    
    Args:
        df_actuals: Actuals dataframe
        df_predicted: Predictions dataframe  
        df_forecast: Forecast dataframe
        
    Returns:
        True if all validations pass
    """
    logger.info("Validating dataframe structures")
    
    validation_passed = True
    
    for name, df in [('df_actuals_yr', df_actuals), ('df_predicted_yr', df_predicted), ('df_forecast_30', df_forecast)]:
        if df.empty:
            logger.error(f"{name} is empty")
            validation_passed = False
            continue
            
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            logger.error(f"{name} missing columns: {missing_cols}")
            validation_passed = False
        else:
            logger.info(f"{name} validation passed - contains all required columns")
    
    return validation_passed

def save_dataframes(df_actuals: pd.DataFrame, df_predicted: pd.DataFrame, 
                   df_forecast: pd.DataFrame, df_metrics: pd.DataFrame) -> bool:
    """
    Save all dataframes as CSV files to the upload directory.
    
    Args:
        df_actuals: Actuals dataframe
        df_predicted: Predictions dataframe
        df_forecast: Forecast dataframe
        df_metrics: Metrics dataframe
        
    Returns:
        True if all saves successful
    """
    logger.info("Saving dataframes to CSV files")
    
    save_operations = [
        (df_actuals, 'actuals.csv', 'Recent year actuals'),
        (df_predicted, 'predicts.csv', 'Recent year predictions'),  
        (df_forecast, 'forecasts.csv', '30-day forecast'),
        (df_metrics, 'model_metrics.csv', 'Model performance metrics')
    ]
    
    all_successful = True
    
    for df, filename, description in save_operations:
        try:
            if df.empty:
                logger.error(f"Cannot save {description}: DataFrame is empty")
                all_successful = False
                continue
            
            # Reorder columns to match REQUIRED_COLUMNS if this is one of the main dataframes
            if filename in ['actuals.csv', 'predicts.csv', 'forecasts.csv']:
                if set(REQUIRED_COLUMNS).issubset(df.columns):
                    df = df[REQUIRED_COLUMNS]  # Reorder columns
                    logger.info(f"Reordered columns for {description} to match REQUIRED_COLUMNS")
                else:
                    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
                    logger.warning(f"Cannot reorder {description} - missing columns: {missing_cols}")
                
            filepath = os.path.join(DATA_UPLOAD, filename)
            df.to_csv(filepath, index=False, date_format='%Y-%m-%d')
            logger.info(f"Saved {description} to {filepath} ({len(df)} rows)")
            
        except Exception as e:
            logger.error(f"Error saving {description}: {e}")
            all_successful = False
    
    return all_successful

def main() -> None:
    """
    Main execution function for the daily forecast process.
    """
    logger.info("=" * 60)
    logger.info("DAILY LSTM FORECAST PROCESS STARTED")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    try:
        # Step 1: Check if market is open
        is_open, reason = is_market_open_today()
        if not is_open:
            logger.info(f"Skipping forecast: {reason}")
            return
        
        # Step 2: Load NYSE calendar
        logger.info("Loading NYSE calendar")
        nyse = mcal.get_calendar('NYSE')
        
        # Step 3: Update FRED data
        logger.info("Fetching latest data from FRED")
        df_data = update_fred_data()
        
        if df_data.empty:
            raise Exception("Failed to fetch data from FRED or no data returned")
        
        logger.info(f"Fetched {len(df_data)} rows of data from FRED")
        
        # Step 4: Get recent year data
        df_actuals_yr = get_recent_data(df_data, DAYS_TO_KEEP)
        
        if df_actuals_yr.empty:
            raise Exception("No recent data available")
        
        # Step 5: Load pre-trained model and scaler
        model, scaler = load_model_and_scaler()
        if model is None or scaler is None:
            raise Exception("Failed to load model or scaler")
        
        # Step 6: Generate predictions
        df_predicted_yr = generate_predictions(model, scaler, df_actuals_yr)
        
        if df_predicted_yr.empty:
            raise Exception("Failed to generate predictions")
        
        # Step 7: Calculate model metrics
        df_model_metrics = calculate_model_metrics(df_actuals_yr, df_predicted_yr)
        
        if df_model_metrics.empty:
            raise Exception("Failed to calculate model metrics")
        
        # Step 8: Generate 30-day forecast
        df_forecast_30 = generate_forecast(model, scaler, df_data)
        
        if df_forecast_30.empty:
            raise Exception("Failed to generate forecast")
        
        # Step 9: Validate dataframes
        if not validate_dataframes(df_actuals_yr, df_predicted_yr, df_forecast_30):
            raise Exception("Dataframe validation failed")
        
        # Step 10: Save all dataframes
        if not save_dataframes(df_actuals_yr, df_predicted_yr, df_forecast_30, df_model_metrics):
            raise Exception("Failed to save one or more dataframes")
        
        # Success!
        logger.info("=" * 60)
        logger.info("DAILY LSTM FORECAST PROCESS COMPLETED SUCCESSFULLY")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
    except Exception as e:
        error_msg = f"Daily forecast process failed: {str(e)}"
        detailed_error = traceback.format_exc()
        
        logger.error("=" * 60)
        logger.error("DAILY LSTM FORECAST PROCESS FAILED")
        logger.error(f"Error: {error_msg}")
        logger.error(f"Detailed traceback:\n{detailed_error}")
        logger.error("=" * 60)
        
        # Send failure notification
        send_failure_notification(error_msg, detailed_error)
        
        # Exit with error code
        sys.exit(1)


if __name__ == "__main__":
    main()