import os
from dotenv import load_dotenv
import pandas as pd
from azure.storage.blob import BlobServiceClient

# Load .env file
load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
if AZURE_STORAGE_CONNECTION_STRING is None:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found! Check your .env file location and spelling.")

# Get Azure Blob Storage connection string
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

# Specify the files and paths
csv_dir = 'data/output'
files = [
    'forecasted_data.csv',
    'historical_data.csv',
    'model_metrics.csv',
    'predicted_data.csv'
]
parquet_files = [f.replace('.csv', '.parquet') for f in files]
parquet_dir = csv_dir  # Saving Parquet files in the same directory

# Convert CSV to Parquet
for file in files:
    csv_path = os.path.join(csv_dir, file)
    parquet_path = os.path.join(parquet_dir, file.replace('.csv', '.parquet'))
    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path, index=False)
    print(f"Converted {csv_path} to {parquet_path}")

# Upload Parquet files to Azure Blob Storage
container_name = 'data'
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

for file in parquet_files:
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=file)
    parquet_path = os.path.join(parquet_dir, file)
    with open(parquet_path, 'rb') as data:
        blob_client.upload_blob(data, overwrite=True)
    print(f"Uploaded {file} to Azure container '{container_name}'")

print("All files converted and uploaded successfully.")