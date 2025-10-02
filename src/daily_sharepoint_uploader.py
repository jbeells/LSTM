#!/usr/bin/env python3
"""
SharePoint CSV Uploader Script
==============================

Uploads CSV files from data/upload to SharePoint document library, overwriting existing files.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import requests
from requests.auth import HTTPBasicAuth
from requests.exceptions import RequestException
import json

# Setup logging to match your existing pattern
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_sharepoint_auth_headers():
    """Get authentication headers for SharePoint API calls"""
    username = os.getenv('SHAREPOINT_USERNAME')
    password = os.getenv('SHAREPOINT_PASSWORD')
    
    if not username or not password:
        raise ValueError("Missing SharePoint credentials: SHAREPOINT_USERNAME and SHAREPOINT_PASSWORD required")
    
    # Create session with basic auth
    session = requests.Session()
    session.auth = HTTPBasicAuth(username, password)
    session.headers.update({
        'Accept': 'application/json;odata=verbose',
        'Content-Type': 'application/json;odata=verbose'
    })
    
    return session

def get_form_digest(session, site_url):
    """Get form digest value required for SharePoint POST operations"""
    try:
        digest_url = f"{site_url}/_api/contextinfo"
        response = session.post(digest_url)
        response.raise_for_status()
        
        digest_data = response.json()
        return digest_data['d']['GetContextWebInformation']['FormDigestValue']
    
    except RequestException as e:
        logger.error(f"Failed to get form digest: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Failed to get form digest: {e}")
        raise

def upload_file_to_sharepoint(session, site_url, folder_path, file_path, form_digest):
    """Upload a single file to SharePoint, overwriting if it exists"""
    try:
        file_name = file_path.name
        
        # Read file content
        with open(file_path, 'rb') as file:
            file_content = file.read()
        
        # SharePoint REST API endpoint for file upload
        # This will overwrite existing files automatically
        upload_url = f"{site_url}/_api/web/GetFolderByServerRelativeUrl('{folder_path}')/Files/add(url='{file_name}',overwrite=true)"
        
        # Headers for file upload
        headers = {
            'Accept': 'application/json;odata=verbose',
            'X-RequestDigest': form_digest,
            'Content-Type': 'application/octet-stream'
        }
        
        response = session.post(upload_url, data=file_content, headers=headers)
        response.raise_for_status()
        
        logger.info(f"Successfully uploaded {file_name} to SharePoint")
        return True
        
    except RequestException as e:
        logger.error(f"Failed to upload {file_path.name}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Failed to upload {file_path.name}: {e}")
        raise

def upload_csv_files():
    """Upload all CSV files from data/upload to SharePoint"""
    
    try:
        # Get SharePoint configuration from environment
        site_url = os.getenv('SHAREPOINT_SITE_URL')  # e.g., https://jeanalytics.sharepoint.com/sites/LSTM
        folder_path = os.getenv('SHAREPOINT_FOLDER_PATH', '/sites/LSTM/Shared Documents')  # Default to Shared Documents
        
        if not site_url:
            raise ValueError("Missing SHAREPOINT_SITE_URL in environment variables")
        
        # Initialize SharePoint session
        session = get_sharepoint_auth_headers()
        
        # Get form digest for authentication
        form_digest = get_form_digest(session, site_url)
        
        # Path to upload directory
        upload_dir = Path(os.path.dirname(__file__)) / 'data' / 'upload'
        
        if not upload_dir.exists():
            raise FileNotFoundError(f"Upload directory not found: {upload_dir}")
        
        # Find all CSV files
        csv_files = list(upload_dir.glob('*.csv'))
        
        if not csv_files:
            logger.warning("No CSV files found to upload")
            return True
        
        # Upload each file (this will overwrite existing files)
        for csv_file in csv_files:
            try:
                upload_file_to_sharepoint(session, site_url, folder_path, csv_file, form_digest)
                
            except Exception as e:
                logger.error(f"Failed to upload {csv_file.name}: {e}")
                raise
        
        logger.info(f"Successfully uploaded {len(csv_files)} files to SharePoint")
        return True
        
    except Exception as e:
        logger.error(f"SharePoint upload process failed: {e}")
        return False

if __name__ == "__main__":
    success = upload_csv_files()
    sys.exit(0 if success else 1)
