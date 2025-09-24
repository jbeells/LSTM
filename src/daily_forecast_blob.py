import os
import io
import json
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import pandas_market_calendars as mcal
from fredapi import Fred

# Load environment variables
load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

# Blob file paths (CSV format)
HISTORICAL_DATA_BLOB = "data/historical_data.csv"
PREDICTED_DATA_BLOB = "data/predicted_data.csv"
METRICS_BLOB = "data/metrics.csv"

# Utility functions
def get_blob_service_client():
    return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

def upload_to_blob(data, blob_name: str, file_format="csv", logger=None):
    """Upload DataFrame or data to Azure Blob Storage."""
    try:
        blob_service_client = get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME, blob=blob_name
        )

        if file_format == "csv":
            buffer = io.StringIO()
            data.to_csv(buffer, index=False)
            content = buffer.getvalue().encode("utf-8")
        elif file_format == "json":
            content = json.dumps(data).encode("utf-8")
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        blob_client.upload_blob(content, overwrite=True)

        if logger:
            logger.info(
                f"Uploaded {len(data) if hasattr(data, '__len__') else 'data'} records to {blob_name}"
            )
    except Exception as e:
        msg = f"Error uploading to blob {blob_name}: {e}"
        if logger:
            logger.error(msg)
        raise

def download_from_blob(blob_name: str, file_format="csv", logger=None):
    """Download data from Azure Blob Storage."""
    try:
        blob_service_client = get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME, blob=blob_name
        )

        if not blob_client.exists():
            return None

        content = blob_client.download_blob().readall()

        if file_format == "csv":
            return pd.read_csv(io.BytesIO(content))
        elif file_format == "json":
            return json.loads(content.decode("utf-8"))
        else:
            raise ValueError(f"Unsupported format: {file_format}")
    except Exception as e:
        if logger:
            logger.warning(f"Could not download {blob_name}: {e}")
        return None

def append_to_blob_csv(new_data: pd.DataFrame, blob_name: str, logger=None):
    """Append data to existing CSV file in blob storage."""
    try:
        existing_data = download_from_blob(blob_name, "csv", logger)

        if existing_data is not None:
            if "Date" in new_data.columns and "Date" in existing_data.columns:
                existing_data["Date"] = pd.to_datetime(existing_data["Date"])
                new_data["Date"] = pd.to_datetime(new_data["Date"])

                new_dates = new_data["Date"].isin(existing_data["Date"])
                new_data_filtered = new_data[~new_dates]

                if len(new_data_filtered) == 0:
                    if logger:
                        logger.info(f"No new data to append to {blob_name}")
                    return

                combined_data = pd.concat(
                    [existing_data, new_data_filtered], ignore_index=True
                )
            else:
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            combined_data = new_data

        upload_to_blob(combined_data, blob_name, "csv", logger)
    except Exception as e:
        msg = f"Error appending to blob {blob_name}: {e}"
        if logger:
            logger.error(msg)
        raise

# Forecast orchestration
def should_run_forecast(run_date, logger=None):
    """Check if forecast already exists for the run date."""
    forecast_blob = f"forecasts/forecasted_{run_date}.csv"
    existing_forecast = download_from_blob(forecast_blob, "csv", logger)
    return existing_forecast is None

def catch_up_missing_data(logger=None):
    """Check for missing historical data and fetch if needed."""
    historical_data = download_from_blob(HISTORICAL_DATA_BLOB, "csv", logger)

    if historical_data is None or historical_data.empty:
        if logger:
            logger.info("No historical data found in blob. Skipping catch-up.")
        return

    historical_data["Date"] = pd.to_datetime(historical_data["Date"])
    last_date = historical_data["Date"].max()

    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=last_date + timedelta(days=1), end_date=datetime.today())
    trading_days = mcal.date_range(schedule, frequency="1D")

    if len(trading_days) > 0:
        if logger:
            logger.info(f"Fetching missing {len(trading_days)} trading days of data")

        fred = Fred(api_key=os.getenv("FRED_API_KEY"))
        new_data = []
        for day in trading_days:
            try:
                value = fred.get_series("DGS10", day, day)
                new_data.append({"Date": day, "DGS10": value})
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to fetch data for {day}: {e}")

        if new_data:
            new_data_df = pd.DataFrame(new_data)
            append_to_blob_csv(new_data_df, HISTORICAL_DATA_BLOB, logger)

def perform_health_check(logger=None):
    """Check if data blobs are present and valid."""
    health = {}
    for blob_name in [HISTORICAL_DATA_BLOB, PREDICTED_DATA_BLOB, METRICS_BLOB]:
        try:
            data = download_from_blob(blob_name, "csv", logger)
            health[blob_name] = data is not None and not data.empty
        except Exception:
            health[blob_name] = False
    return health

def run_forecast(run_date, logger=None):
    """Main forecasting routine."""
    if not should_run_forecast(run_date, logger):
        if logger:
            logger.info(f"Forecast already exists for {run_date}, skipping run.")
        return

    # Load historical data
    historical_data = download_from_blob(HISTORICAL_DATA_BLOB, "csv", logger)
    if historical_data is None or historical_data.empty:
        if logger:
            logger.error("No historical data available, cannot run forecast.")
        return

    # Dummy model prediction for demonstration
    forecast = pd.DataFrame(
        {
            "Date": [pd.to_datetime(run_date)],
            "Forecast": [np.random.rand()],
        }
    )

    # Save forecast
    forecast_blob = f"forecasts/forecasted_{run_date}.csv"
    upload_to_blob(forecast, forecast_blob, "csv", logger)

    # Append prediction
    append_to_blob_csv(forecast, PREDICTED_DATA_BLOB, logger)

    # Dummy metrics
    metrics = pd.DataFrame(
        {
            "Date": [pd.to_datetime(run_date)],
            "RMSE": [mean_squared_error([0], [forecast["Forecast"].iloc[0]], squared=False)],
            "MAE": [mean_absolute_error([0], [forecast["Forecast"].iloc[0]])],
            "R2": [r2_score([0], [forecast["Forecast"].iloc[0]])],
        }
    )
    append_to_blob_csv(metrics, METRICS_BLOB, logger)

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("daily_forecast_blob")

    run_date = datetime.today().strftime("%Y-%m-%d")

    logger.info(f"Starting daily forecast run for {run_date}")

    # Catch-up
    catch_up_missing_data(logger)

    # Health check
    health = perform_health_check(logger)
    logger.info(f"Health check: {health}")

    # Forecast
    run_forecast(run_date, logger)

if __name__ == "__main__":
    main()
