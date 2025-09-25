"""
DIAGNOSTIC VERSION - Daily forecasting script with enhanced debugging
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
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_blob_service_client():
    """Get Azure Blob Service Client with enhanced debugging."""
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is required")
    
    client = BlobServiceClient.from_connection_string(connection_string)
    return client

def list_container_contents(logger=None):
    """List all blobs in the container for debugging."""
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        
        if logger:
            logger.info(f"=== CONTAINER CONTENTS: {AZURE_CONTAINER_NAME} ===")
        
        blobs = list(container_client.list_blobs())
        for blob in blobs:
            if logger:
                logger.info(f"  - {blob.name} (size: {blob.size} bytes, modified: {blob.last_modified})")
        
        if not blobs:
            if logger:
                logger.info("  Container is empty")
                
        return blobs
        
    except Exception as e:
        if logger:
            logger.error(f"Error listing container contents: {e}")
        return []

def upload_to_blob(data, blob_name: str, file_format='csv', logger=None):
    """Upload DataFrame or data to Azure Blob Storage with enhanced debugging."""
    try:
        if logger:
            logger.info(f"=== UPLOAD STARTING ===")
            logger.info(f"Container: {AZURE_CONTAINER_NAME}")
            logger.info(f"Blob name: {blob_name}")
            logger.info(f"Format: {file_format}")
            
            if hasattr(data, 'shape'):
                logger.info(f"Data shape: {data.shape}")
            elif hasattr(data, '__len__'):
                logger.info(f"Data length: {len(data)}")

        blob_service_client = get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=blob_name
        )

        # Convert to bytes
        if file_format == 'csv':
            buffer = io.StringIO()
            data.to_csv(buffer, index=False)
            content = buffer.getvalue()
            content_bytes = content.encode('utf-8')
            
            if logger:
                logger.info(f"CSV content length: {len(content)} chars, {len(content_bytes)} bytes")
                logger.info(f"First 200 chars: {content[:200]}")
                
        elif file_format == 'json':
            content = json.dumps(data, indent=2)
            content_bytes = content.encode('utf-8')
            
            if logger:
                logger.info(f"JSON content length: {len(content)} chars, {len(content_bytes)} bytes")
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        # Upload with explicit parameters
        if logger:
            logger.info(f"Uploading {len(content_bytes)} bytes to {AZURE_CONTAINER_NAME}/{blob_name}")
            
        blob_client.upload_blob(
            data=content_bytes, 
            overwrite=True,
            content_type='text/csv' if file_format == 'csv' else 'application/json'
        )

        if logger:
            logger.info(f"✓ Upload completed successfully")
            
            # Verify the upload by checking if blob exists
            if blob_client.exists():
                properties = blob_client.get_blob_properties()
                logger.info(f"✓ Blob verified - Size: {properties.size} bytes, Last modified: {properties.last_modified}")
            else:
                logger.error(f"✗ Upload failed - Blob does not exist after upload")

    except Exception as e:
        error_msg = f"Error uploading to blob {blob_name}: {e}"
        if logger:
            logger.error(error_msg)
        raise

def download_from_blob(blob_name: str, file_format='csv', logger=None):
    """Download data from Azure Blob Storage with enhanced debugging."""
    try:
        if logger:
            logger.info(f"=== DOWNLOAD STARTING ===")
            logger.info(f"Container: {AZURE_CONTAINER_NAME}")
            logger.info(f"Blob name: {blob_name}")

        blob_service_client = get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=blob_name
        )

        if not blob_client.exists():
            if logger:
                logger.info(f"Blob {blob_name} does not exist")
            return None

        properties = blob_client.get_blob_properties()
        if logger:
            logger.info(f"Blob found - Size: {properties.size} bytes, Last modified: {properties.last_modified}")

        content = blob_client.download_blob().readall()
        
        if logger:
            logger.info(f"Downloaded {len(content)} bytes")

        if file_format == 'csv':
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            if logger:
                logger.info(f"Parsed CSV - Shape: {df.shape}")
                if len(df) > 0:
                    logger.info(f"Columns: {list(df.columns)}")
                    logger.info(f"First few rows:\n{df.head()}")
            return df
        elif file_format == 'json':
            data = json.loads(content.decode('utf-8'))
            if logger:
                logger.info(f"Parsed JSON with {len(data) if hasattr(data, '__len__') else 'unknown'} items")
            return data
        else:
            raise ValueError(f"Unsupported format: {file_format}")

    except Exception as e:
        if logger:
            logger.warning(f"Could not download {blob_name}: {e}")
        return None

def test_basic_operations(logger=None):
    """Test basic blob operations."""
    try:
        if logger:
            logger.info("=== TESTING BASIC BLOB OPERATIONS ===")
        
        # Test 1: List container contents before
        logger.info("BEFORE TEST:")
        list_container_contents(logger)
        
        # Test 2: Upload a simple test file
        test_data = pd.DataFrame({
            'Date': ['2024-09-25'],
            'SP500': [5000.0],
            'VIXCLS': [20.0],
            'DJIA': [40000.0],
            'HY_BOND_IDX': [100.0]
        })
        
        test_blob_name = 'test_upload.csv'
        if logger:
            logger.info(f"Uploading test data: {test_blob_name}")
        
        upload_to_blob(test_data, test_blob_name, 'csv', logger)
        
        # Test 3: List container contents after upload
        logger.info("AFTER TEST UPLOAD:")
        list_container_contents(logger)
        
        # Test 4: Download the test file
        if logger:
            logger.info(f"Downloading test data: {test_blob_name}")
        
        downloaded_data = download_from_blob(test_blob_name, 'csv', logger)
        
        if downloaded_data is not None:
            if logger:
                logger.info("✓ Test upload/download successful")
        else:
            if logger:
                logger.error("✗ Test download failed")
                
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Basic operations test failed: {e}")
        return False

def main():
    """Main daily forecast execution with enhanced debugging."""
    logger = setup_logging()
    run_date = datetime.date.today().strftime('%Y-%m-%d')

    logger.info(f"=" * 80)
    logger.info(f"DAILY FORECAST DIAGNOSTIC VERSION - {run_date}")
    logger.info(f"Using Azure Blob Storage container: {AZURE_CONTAINER_NAME}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"=" * 80)

    # Environment variable check
    logger.info("=== ENVIRONMENT VARIABLES ===")
    logger.info(f"AZURE_CONTAINER_NAME: {AZURE_CONTAINER_NAME}")
    logger.info(f"AZURE_STORAGE_CONNECTION_STRING: {'SET' if os.getenv('AZURE_STORAGE_CONNECTION_STRING') else 'NOT SET'}")
    logger.info(f"FRED_API_KEY: {'SET' if os.getenv('FRED_API_KEY') else 'NOT SET'}")

    try:
        # Test basic blob operations first
        logger.info("=== TESTING BLOB OPERATIONS ===")
        test_success = test_basic_operations(logger)
        
        if not test_success:
            raise Exception("Basic blob operations test failed")

        # List current container state
        logger.info("=== CURRENT CONTAINER STATE ===")
        existing_blobs = list_container_contents(logger)
        
        # Check for required model files
        model_exists = any(blob.name == MODEL_BLOB for blob in existing_blobs)
        scaler_exists = any(blob.name == SCALER_BLOB for blob in existing_blobs)
        
        logger.info(f"Model file ({MODEL_BLOB}) exists: {model_exists}")
        logger.info(f"Scaler file ({SCALER_BLOB}) exists: {scaler_exists}")
        
        if not model_exists or not scaler_exists:
            logger.error("Required model files are missing from blob storage!")
            logger.error("Please ensure lstm_model.keras and scaler.pkl are in the container root")
            return
        
        # Try a simple health check upload
        logger.info("=== UPLOADING HEALTH CHECK ===")
        health_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'status': 'diagnostic_run',
            'container': AZURE_CONTAINER_NAME,
            'blobs_found': [blob.name for blob in existing_blobs]
        }
        
        upload_to_blob(health_data, HEALTH_CHECK_BLOB, 'json', logger)
        
        # List container after health check upload
        logger.info("=== AFTER HEALTH CHECK UPLOAD ===")
        list_container_contents(logger)
        
        # Try downloading existing historical data
        logger.info("=== TESTING HISTORICAL DATA DOWNLOAD ===")
        historical_data = download_from_blob(HISTORICAL_DATA_BLOB, 'csv', logger)
        
        if historical_data is not None:
            logger.info(f"Historical data found: {len(historical_data)} rows")
            logger.info(f"Date range: {historical_data['Date'].min()} to {historical_data['Date'].max()}")
        else:
            logger.info("No historical data found - this might be the first run")

        logger.info("=== DIAGNOSTIC COMPLETED ===")
        logger.info("Check the container contents above to see if files were uploaded")

    except Exception as e:
        logger.error(f"Diagnostic failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()