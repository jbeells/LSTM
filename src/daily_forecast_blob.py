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
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME', 'lstm')
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
            # Remove file handler - only use console in GitHub Actions
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_blob_service_client():
    """Get Azure Blob Service Client."""
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is required")
    return BlobServiceClient.from_connection_string(connection_string)

def ensure_container_exists(container_name: str, logger=None):
    """Ensure the specified container exists."""
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(container_name)
        
        # Just verify it exists - don't try to create
        if not container_client.exists():
            if logger:
                logger.warning(f"Container {container_name} does not exist")
        else:
            if logger:
                logger.info(f"Container {container_name} exists")
                
    except Exception as e:
        if logger:
            logger.warning(f"Could not verify container {container_name}: {e}")

def upload_to_blob(data, blob_name: str, file_format='csv', logger=None):
    """Upload DataFrame or data to Azure Blob Storage."""
    try:
        blob_service_client = get_blob_service_client()
        
        # Use absolute blob path - no folders
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=blob_name  # Use exact blob name without any path manipulation
        )

        # Convert to bytes
        if file_format == 'csv':
            buffer = io.StringIO()
            data.to_csv(buffer, index=False)
            content = buffer.getvalue().encode('utf-8')
        elif file_format == 'json':
            content = json.dumps(data, indent=2).encode('utf-8')
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        # Upload directly - overwrite existing
        blob_client.upload_blob(content, overwrite=True)

        if logger:
            logger.info(f"Successfully uploaded to blob: {AZURE_CONTAINER_NAME}/{blob_name}")

    except Exception as e:
        error_msg = f"Error uploading to blob {blob_name}: {e}"
        if logger:
            logger.error(error_msg)
        raise

def download_from_blob(blob_name: str, file_format='csv', logger=None):
    """Download data from Azure Blob Storage."""
    try:
        blob_service_client = get_blob_service_client()
        
        # Use absolute blob path
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=blob_name  # Use exact blob name
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
    """Load model and scaler from Azure Blob Storage."""
    try:
        blob_service_client = get_blob_service_client()

        # Download model with explicit path
        if logger:
            logger.info(f"Loading model from: {AZURE_CONTAINER_NAME}/{MODEL_BLOB}")
            
        model_blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=MODEL_BLOB  # Direct blob name - no path manipulation
        )

        if not model_blob_client.exists():
            raise Exception(f"Model blob {MODEL_BLOB} does not exist in container {AZURE_CONTAINER_NAME}")

        model_content = model_blob_client.download_blob().readall()

        # Use a unique temp file name to avoid conflicts
        temp_dir = tempfile.mkdtemp()
        temp_model_path = os.path.join(temp_dir, 'model.keras')
        
        with open(temp_model_path, 'wb') as temp_file:
            temp_file.write(model_content)

        model = tf.keras.models.load_model(temp_model_path)

        # Clean up temp files
        os.remove(temp_model_path)
        os.rmdir(temp_dir)

        # Download scaler
        if logger:
            logger.info(f"Loading scaler from: {AZURE_CONTAINER_NAME}/{SCALER_BLOB}")
            
        scaler_blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=SCALER_BLOB  # Direct blob name
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

# [Keep all other functions the same as in your original file, but apply the same pattern:
# - Remove file logging
# - Use direct blob names without path manipulation
# - Add explicit container/blob logging]

def main():
    """Main daily forecast execution with enhanced robustness."""
    # Setup logging - no file logging in GitHub Actions
    logger = setup_logging()
    run_date = datetime.date.today().strftime('%Y-%m-%d')

    logger.info(f"=" * 80)
    logger.info(f"DAILY FORECAST STARTED - {run_date}")
    logger.info(f"Using Azure Blob Storage container: {AZURE_CONTAINER_NAME}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"=" * 80)

    # Rest of your main function...
    # [Keep the same logic but ensure all blob operations use direct paths]

if __name__ == '__main__':
    main()