#!/usr/bin/env python3
"""
Daily Blob Uploader Script
==========================

Uploads CSV files from data/upload to Azure Blob Storage after successful forecast completion.
"""

import os
import sys
import logging
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from pathlib import Path

# Setup logging to match your existing pattern
logger = logging.getLogger(__name__)

def upload_csv_files():
    """Upload all CSV files from data/upload to Azure Blob Storage"""
    
    try:
        # Get Azure credentials from environment
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        container_name = os.getenv('AZURE_CONTAINER_NAME')
        
        if not connection_string or not container_name:
            raise ValueError("Missing Azure credentials in environment variables")
        
        # Initialize blob client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Path to upload directory
        upload_dir = Path(os.path.dirname(os.path.dirname(__file__))) / 'data' / 'upload'
        
        if not upload_dir.exists():
            raise FileNotFoundError(f"Upload directory not found: {upload_dir}")
        
        # Find all CSV files
        csv_files = list(upload_dir.glob('*.csv'))
        
        if not csv_files:
            logger.warning("No CSV files found to upload")
            return True
        
        # Upload each file
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for csv_file in csv_files:
            try:
                # Create blob name with timestamp
                blob_name = f"{csv_file.name}" # f"{timestamp}/{csv_file.name}"
                
                # Upload file
                with open(csv_file, 'rb') as data:
                    blob_client = blob_service_client.get_blob_client(
                        container=container_name, 
                        blob=blob_name
                    )
                    blob_client.upload_blob(data, overwrite=True)
                
                logger.info(f"Uploaded {csv_file.name} to {blob_name}")
                
            except Exception as e:
                logger.error(f"Failed to upload {csv_file.name}: {e}")
                raise
        
        logger.info(f"Successfully uploaded {len(csv_files)} files to Azure Blob Storage")
        return True
        
    except Exception as e:
        logger.error(f"Blob upload process failed: {e}")
        return False

if __name__ == "__main__":
    success = upload_csv_files()
    sys.exit(0 if success else 1)