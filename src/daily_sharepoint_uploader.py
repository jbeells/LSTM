#!/usr/bin/env python3
"""
SharePoint CSV Uploader Script - App Registration Version
========================================================

Uploads CSV files from data/upload to SharePoint document library using App Registration.
This method works with modern SharePoint Online authentication requirements.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import requests
from requests.exceptions import RequestException
import json
import urllib.parse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_access_token():
    """Get access token using App Registration (Client Credentials Flow)"""
    tenant_id = os.getenv('SHAREPOINT_TENANT_ID')
    client_id = os.getenv('SHAREPOINT_CLIENT_ID')
    client_secret = os.getenv('SHAREPOINT_CLIENT_SECRET')
    
    if not all([tenant_id, client_id, client_secret]):
        raise ValueError("Missing required app registration credentials: SHAREPOINT_TENANT_ID, SHAREPOINT_CLIENT_ID, SHAREPOINT_CLIENT_SECRET")
    
    # Microsoft Graph token endpoint
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    
    # Get SharePoint resource from site URL
    site_url = os.getenv('SHAREPOINT_SITE_URL')
    if not site_url:
        raise ValueError("Missing SHAREPOINT_SITE_URL")
    
    # Extract tenant domain from SharePoint URL
    from urllib.parse import urlparse
    parsed_url = urlparse(site_url)
    sharepoint_resource = f"https://{parsed_url.hostname}"
    
    token_data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
        'scope': f'{sharepoint_resource}/.default'
    }
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    try:
        response = requests.post(token_url, data=token_data, headers=headers)
        response.raise_for_status()
        
        token_info = response.json()
        access_token = token_info['access_token']
        
        logger.info("Successfully obtained access token")
        return access_token
        
    except RequestException as e:
        logger.error(f"Failed to get access token: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        raise

def create_authenticated_session(access_token):
    """Create requests session with Bearer token authentication"""
    session = requests.Session()
    session.headers.update({
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json;odata=verbose',
        'Content-Type': 'application/json;odata=verbose'
    })
    return session

def get_form_digest(session, site_url):
    """Get form digest value for SharePoint POST operations"""
    try:
        digest_url = f"{site_url}/_api/contextinfo"
        
        response = session.post(digest_url)
        response.raise_for_status()
        
        digest_data = response.json()
        form_digest = digest_data['d']['GetContextWebInformation']['FormDigestValue']
        
        logger.info("Successfully obtained form digest")
        return form_digest
        
    except RequestException as e:
        logger.error(f"Failed to get form digest: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        raise

def upload_file_to_sharepoint(session, site_url, folder_path, file_path, form_digest):
    """Upload a single file to SharePoint, overwriting if it exists"""
    try:
        file_name = file_path.name
        
        # Read file content
        with open(file_path, 'rb') as file:
            file_content = file.read()
        
        # URL encode the folder path properly
        encoded_folder_path = urllib.parse.quote(folder_path, safe='/')
        
        # SharePoint REST API endpoint for file upload
        upload_url = f"{site_url}/_api/web/GetFolderByServerRelativeUrl('{encoded_folder_path}')/Files/add(url='{file_name}',overwrite=true)"
        
        # Create new session for file upload with different headers
        upload_session = requests.Session()
        upload_session.headers.update({
            'Authorization': session.headers['Authorization'],  # Keep the Bearer token
            'Accept': 'application/json;odata=verbose',
            'X-RequestDigest': form_digest,
            'Content-Type': 'application/octet-stream'
        })
        
        logger.info(f"Uploading {file_name} ({len(file_content)} bytes)")
        
        response = upload_session.post(upload_url, data=file_content)
        response.raise_for_status()
        
        logger.info(f"Successfully uploaded {file_name} to SharePoint")
        return True
        
    except RequestException as e:
        logger.error(f"Failed to upload {file_path.name}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
            
            if e.response.status_code == 404:
                logger.error("Folder not found. Please check the SHAREPOINT_FOLDER_PATH")
            elif e.response.status_code == 403:
                logger.error("Access denied. Check app permissions in SharePoint")
        raise

def test_folder_access(session, site_url, folder_path):
    """Test if the target folder exists and is accessible"""
    try:
        encoded_folder_path = urllib.parse.quote(folder_path, safe='/')
        test_url = f"{site_url}/_api/web/GetFolderByServerRelativeUrl('{encoded_folder_path}')"
        
        response = session.get(test_url)
        if response.status_code == 200:
            logger.info(f"✓ Folder exists and is accessible: {folder_path}")
            return True
        elif response.status_code == 404:
            logger.error(f"✗ Folder not found: {folder_path}")
            return False
        else:
            logger.warning(f"Folder access test returned status {response.status_code}")
            return True  # Proceed anyway
            
    except Exception as e:
        logger.warning(f"Could not test folder access: {e}")
        return True  # Proceed anyway

def upload_csv_files():
    """Upload all CSV files from data/upload to SharePoint"""
    
    try:
        # Get SharePoint configuration from environment
        site_url = os.getenv('SHAREPOINT_SITE_URL')
        folder_path = os.getenv('SHAREPOINT_FOLDER_PATH', '/Shared Documents')
        
        if not site_url:
            raise ValueError("Missing SHAREPOINT_SITE_URL in environment variables")
        
        logger.info(f"Connecting to SharePoint site: {site_url}")
        logger.info(f"Target folder: {folder_path}")
        
        # Get access token and create authenticated session
        access_token = get_access_token()
        session = create_authenticated_session(access_token)
        
        # Test folder access
        test_folder_access(session, site_url, folder_path)
        
        # Get form digest for uploads
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
        
        logger.info(f"Found {len(csv_files)} CSV files to upload")
        
        # Upload each file
        success_count = 0
        for csv_file in csv_files:
            try:
                upload_file_to_sharepoint(session, site_url, folder_path, csv_file, form_digest)
                success_count += 1
                
            except Exception as e:
                logger.error(f"Failed to upload {csv_file.name}: {e}")
                # Continue with other files rather than stopping completely
                continue
        
        if success_count == len(csv_files):
            logger.info(f"Successfully uploaded all {success_count} files to SharePoint")
            return True
        elif success_count > 0:
            logger.warning(f"Uploaded {success_count} out of {len(csv_files)} files")
            return True
        else:
            logger.error("Failed to upload any files")
            return False
        
    except Exception as e:
        logger.error(f"SharePoint upload process failed: {e}")
        logger.error("Setup instructions:")
        logger.error("1. Create an App Registration in Azure AD")
        logger.error("2. Grant Sites.ReadWrite.All permissions")
        logger.error("3. Add the app to your SharePoint site with edit permissions")
        logger.error("4. Set the required environment variables:")
        logger.error("   - SHAREPOINT_TENANT_ID")
        logger.error("   - SHAREPOINT_CLIENT_ID") 
        logger.error("   - SHAREPOINT_CLIENT_SECRET")
        return False

if __name__ == "__main__":
    success = upload_csv_files()
    sys.exit(0 if success else 1)
