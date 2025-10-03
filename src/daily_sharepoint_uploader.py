#!/usr/bin/env python3
"""
SharePoint CSV Uploader Script - REST API Version
================================================

Alternative approach using SharePoint REST API instead of Microsoft Graph API.
This may work better in some tenant configurations.

Files uploaded:
- actuals.csv (recent year actuals)
- predicts.csv (recent year predictions) 
- forecasts.csv (30-day forecast)
- model_metrics.csv (model performance metrics)

Author: AI Assistant
Date: 2024-10-03
"""

import os
import sys
import logging
import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import urllib.parse
import base64

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(LOG_DIR, exist_ok=True)
    
    log_filename = f"sharepoint_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    logger.info(f"SharePoint uploader logging initialized. Log file: {log_path}")
    return logger

logger = setup_logging()

class SharePointRESTUploader:
    """
    SharePoint uploader using SharePoint REST API with App Registration
    """
    
    def __init__(self):
        """Initialize SharePoint uploader with configuration from environment variables"""
        self.tenant_id = os.getenv('SHAREPOINT_TENANT_ID')
        self.client_id = os.getenv('SHAREPOINT_CLIENT_ID') 
        self.client_secret = os.getenv('SHAREPOINT_CLIENT_SECRET')
        
        # SharePoint site configuration
        self.site_url = os.getenv('SHAREPOINT_SITE_URL', 'https://jeanalytics.sharepoint.com/sites/LSTM')
        self.folder_path = os.getenv('SHAREPOINT_FOLDER_PATH', 'Shared Documents')
        
        # Extract tenant domain from site URL
        if 'sharepoint.com' in self.site_url:
            url_parts = self.site_url.replace('https://', '').split('/')
            self.tenant_domain = url_parts[0]  # tenant.sharepoint.com
            self.site_name = url_parts[-1] if len(url_parts) > 2 else ''
        else:
            raise ValueError(f"Invalid SharePoint site URL: {self.site_url}")
        
        # SharePoint REST API endpoints
        self.rest_base_url = f"{self.site_url}/_api"
        self.token_url = f"https://accounts.accesscontrol.windows.net/{self.tenant_id}/tokens/OAuth/2"
        self.graph_token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        
        self.access_token = None
        
        logger.info(f"SharePoint REST uploader initialized for site: {self.site_url}")
        logger.info(f"Target folder: {self.folder_path}")
        logger.info(f"REST API base: {self.rest_base_url}")
        
    def validate_config(self) -> bool:
        """Validate that all required configuration is present"""
        required_vars = {
            'SHAREPOINT_TENANT_ID': self.tenant_id,
            'SHAREPOINT_CLIENT_ID': self.client_id, 
            'SHAREPOINT_CLIENT_SECRET': self.client_secret
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            logger.error("Please ensure the following secrets are set in GitHub:")
            for var in missing_vars:
                logger.error(f"  - {var}")
            return False
            
        logger.info("All required configuration variables are present")
        return True
        
    def get_sharepoint_access_token(self) -> bool:
        """
        Get SharePoint access token using SharePoint-specific OAuth endpoint
        """
        logger.info("Requesting SharePoint access token")
        
        # SharePoint resource identifier
        resource = f"00000003-0000-0ff1-ce00-000000000000/{self.tenant_domain}@{self.tenant_id}"
        
        token_data = {
            'grant_type': 'client_credentials',
            'client_id': f"{self.client_id}@{self.tenant_id}",
            'client_secret': self.client_secret,
            'resource': resource
        }
        
        try:
            response = requests.post(self.token_url, data=token_data)
            
            if response.status_code == 200:
                token_response = response.json()
                self.access_token = token_response.get('access_token')
                
                if self.access_token:
                    logger.info("Successfully obtained SharePoint access token")
                    return True
                else:
                    logger.error("Access token not found in SharePoint response")
                    return False
            else:
                logger.warning(f"SharePoint token request failed: {response.status_code}")
                logger.warning(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.warning(f"SharePoint token request failed with exception: {e}")
            return False
            
    def get_graph_access_token(self) -> bool:
        """
        Get access token using Microsoft Graph (fallback method)
        """
        logger.info("Requesting Microsoft Graph access token as fallback")
        
        token_data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': f'https://{self.tenant_domain}/AllSites.Write',
            'grant_type': 'client_credentials'
        }
        
        try:
            response = requests.post(self.graph_token_url, data=token_data)
            
            if response.status_code == 200:
                token_response = response.json()
                self.access_token = token_response.get('access_token')
                
                if self.access_token:
                    logger.info("Successfully obtained Graph access token for SharePoint")
                    return True
                else:
                    logger.error("Access token not found in Graph response")
                    return False
            else:
                logger.warning(f"Graph token request failed: {response.status_code}")
                logger.warning(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.warning(f"Graph token request failed with exception: {e}")
            return False
    
    def get_access_token(self) -> bool:
        """
        Try multiple token acquisition methods
        """
        logger.info("Attempting to get access token using multiple methods")
        
        # Method 1: SharePoint-specific token
        if self.get_sharepoint_access_token():
            return True
            
        # Method 2: Graph API token with SharePoint scope
        if self.get_graph_access_token():
            return True
            
        logger.error("All token acquisition methods failed")
        return False
        
    def test_rest_api_access(self) -> bool:
        """
        Test basic REST API access
        """
        logger.info("Testing SharePoint REST API access")
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json;odata=verbose',
            'Content-Type': 'application/json'
        }
        
        try:
            # Test basic site access
            test_url = f"{self.rest_base_url}/web"
            response = requests.get(test_url, headers=headers)
            
            if response.status_code == 200:
                site_info = response.json()
                site_title = site_info.get('d', {}).get('Title', 'Unknown')
                logger.info(f"Successfully accessed site: {site_title}")
                return True
            else:
                logger.error(f"REST API test failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"REST API test failed with exception: {e}")
            return False
            
    def get_folder_info(self) -> Optional[Dict]:
        """
        Get information about the target folder
        """
        logger.info(f"Getting folder information for: {self.folder_path}")
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json;odata=verbose',
            'Content-Type': 'application/json'
        }
        
        try:
            # For "Shared Documents", use the default document library
            if self.folder_path.lower() in ['shared documents', 'documents']:
                folder_url = f"{self.rest_base_url}/web/lists/getbytitle('Documents')"
            else:
                folder_path_encoded = urllib.parse.quote(self.folder_path)
                folder_url = f"{self.rest_base_url}/web/getfolderbyserverrelativeurl('{folder_path_encoded}')"
            
            logger.info(f"Folder URL: {folder_url}")
            response = requests.get(folder_url, headers=headers)
            
            if response.status_code == 200:
                folder_info = response.json()
                logger.info("Successfully accessed target folder/library")
                return folder_info
            else:
                logger.warning(f"Folder access failed: {response.status_code}")
                logger.warning(f"Response: {response.text}")
                return None
                
        except Exception as e:
            logger.warning(f"Folder access failed with exception: {e}")
            return None
            
    def upload_file_rest(self, file_path: str, sharepoint_filename: str) -> bool:
        """
        Upload file using SharePoint REST API
        """
        logger.info(f"Uploading {file_path} as {sharepoint_filename} via REST API")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        try:
            # Read file content
            with open(file_path, 'rb') as file:
                file_content = file.read()
                
            # Base64 encode the file content for REST API
            file_content_b64 = base64.b64encode(file_content).decode('utf-8')
                
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json;odata=verbose',
                'Content-Type': 'application/json'
            }
            
            # Upload to Shared Documents library
            upload_url = f"{self.rest_base_url}/web/lists/getbytitle('Documents')/RootFolder/Files/add(url='{sharepoint_filename}',overwrite=true)"
            
            # For REST API, we need to send the binary content directly
            headers['Content-Type'] = 'application/octet-stream'
            
            response = requests.post(upload_url, headers=headers, data=file_content)
            
            if response.status_code in [200, 201]:
                logger.info(f"Successfully uploaded {sharepoint_filename} via REST API")
                return True
            else:
                logger.error(f"REST API upload failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"REST API upload failed with exception: {e}")
            return False
            
    def upload_csv_files(self) -> bool:
        """
        Upload all CSV files to SharePoint using REST API
        """
        logger.info("Starting CSV files upload to SharePoint via REST API")
        
        # Define files to upload
        csv_files = [
            ('actuals.csv', 'actuals.csv'),
            ('predicts.csv', 'predicts.csv'), 
            ('forecasts.csv', 'forecasts.csv'),
            ('model_metrics.csv', 'model_metrics.csv')
        ]
        
        # Get upload directory path
        project_root = os.path.dirname(os.path.dirname(__file__))
        upload_dir = os.path.join(project_root, 'data', 'upload')
        
        logger.info(f"Looking for CSV files in: {upload_dir}")
        
        # Check if upload directory exists
        if not os.path.exists(upload_dir):
            logger.error(f"Upload directory not found: {upload_dir}")
            return False
            
        # Validate configuration
        if not self.validate_config():
            return False
            
        # Get access token
        if not self.get_access_token():
            return False
            
        # Test REST API access
        if not self.test_rest_api_access():
            logger.error("REST API access test failed")
            return False
            
        # Get folder information
        folder_info = self.get_folder_info()
        if folder_info is None:
            logger.warning("Could not access target folder, but continuing with upload attempt")
            
        # Upload each file
        upload_results = []
        
        for local_filename, sharepoint_filename in csv_files:
            local_file_path = os.path.join(upload_dir, local_filename)
            
            if not os.path.exists(local_file_path):
                logger.warning(f"File not found, skipping: {local_file_path}")
                upload_results.append(False)
                continue
                
            # Check file size
            file_size = os.path.getsize(local_file_path)
            logger.info(f"File size for {local_filename}: {file_size} bytes")
            
            if file_size == 0:
                logger.warning(f"File is empty, skipping: {local_filename}")
                upload_results.append(False)
                continue
                
            # Upload file
            success = self.upload_file_rest(local_file_path, sharepoint_filename)
            upload_results.append(success)
            
        # Report results
        successful_uploads = sum(upload_results)
        total_files = len(csv_files)
        
        logger.info(f"Upload summary: {successful_uploads}/{total_files} files uploaded successfully")
        
        return successful_uploads > 0


def main():
    """
    Main execution function for SharePoint upload
    """
    logger.info("=" * 60)
    logger.info("SHAREPOINT CSV UPLOAD PROCESS STARTED (REST API VERSION)")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    try:
        uploader = SharePointRESTUploader()
        success = uploader.upload_csv_files()
        
        if success:
            logger.info("=" * 60)
            logger.info("SHAREPOINT CSV UPLOAD PROCESS COMPLETED SUCCESSFULLY")
            logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 60)
        else:
            raise Exception("SharePoint upload process failed")
            
    except Exception as e:
        logger.error("=" * 60)
        logger.error("SHAREPOINT CSV UPLOAD PROCESS FAILED")
        logger.error(f"Error: {str(e)}")
        logger.error(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
