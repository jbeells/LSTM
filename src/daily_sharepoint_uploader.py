#!/usr/bin/env python3
"""
SharePoint CSV Uploader Script - Fixed Token Type
================================================

Uses the correct token type and authentication method for SharePoint REST API.
This version addresses the "Token type is not allowed" error.

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

class SharePointUploader:
    """
    SharePoint uploader using Microsoft Graph API with proper authentication
    """
    
    def __init__(self):
        """Initialize SharePoint uploader with configuration from environment variables"""
        self.tenant_id = os.getenv('SHAREPOINT_TENANT_ID')
        self.client_id = os.getenv('SHAREPOINT_CLIENT_ID') 
        self.client_secret = os.getenv('SHAREPOINT_CLIENT_SECRET')
        
        # SharePoint site configuration
        self.site_url = os.getenv('SHAREPOINT_SITE_URL', 'https://jeanalytics.sharepoint.com/sites/LSTM')
        self.folder_path = os.getenv('SHAREPOINT_FOLDER_PATH', 'Shared Documents')
        
        # Extract domain info
        if 'sharepoint.com' in self.site_url:
            url_parts = self.site_url.replace('https://', '').split('/')
            self.tenant_name = url_parts[0].split('.')[0]  # Extract tenant name
            self.hostname = url_parts[0]  # full hostname
            self.site_name = url_parts[-1] if len(url_parts) > 2 else 'root'
        else:
            raise ValueError(f"Invalid SharePoint site URL: {self.site_url}")
        
        # Microsoft Graph API endpoints
        self.graph_base_url = "https://graph.microsoft.com/v1.0"
        self.token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        
        self.access_token = None
        
        logger.info(f"SharePoint uploader initialized")
        logger.info(f"Site: {self.site_url}")
        logger.info(f"Folder: {self.folder_path}")
        logger.info(f"Tenant: {self.tenant_name}, Hostname: {self.hostname}")
        
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
            return False
            
        logger.info("All required configuration variables are present")
        return True
        
    def get_access_token(self) -> bool:
        """
        Get access token using client credentials flow with correct scope
        """
        logger.info("Requesting Microsoft Graph access token")
        
        # Use the standard Graph API scope
        token_data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': 'https://graph.microsoft.com/.default',
            'grant_type': 'client_credentials'
        }
        
        try:
            response = requests.post(self.token_url, data=token_data)
            response.raise_for_status()
            
            token_response = response.json()
            self.access_token = token_response.get('access_token')
            
            if self.access_token:
                # Log token info for debugging (first/last 10 chars only)
                token_preview = f"{self.access_token[:10]}...{self.access_token[-10:]}"
                logger.info(f"Successfully obtained access token: {token_preview}")
                
                # Check token type and scope
                token_type = token_response.get('token_type', 'unknown')
                expires_in = token_response.get('expires_in', 'unknown')
                logger.info(f"Token type: {token_type}, expires in: {expires_in} seconds")
                
                return True
            else:
                logger.error("Access token not found in response")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to obtain access token: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            return False
    
    def find_site_by_url(self) -> Optional[str]:
        """
        Find site using the direct URL method with better error handling
        """
        logger.info("Finding SharePoint site by URL")
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        # Try different URL formats
        url_variations = [
            f"{self.graph_base_url}/sites/{self.hostname}:/sites/{self.site_name}",
            f"{self.graph_base_url}/sites/{self.tenant_name}.sharepoint.com:/sites/{self.site_name}",
            f"{self.graph_base_url}/sites/{self.hostname},/sites/{self.site_name}",
        ]
        
        for i, site_url in enumerate(url_variations, 1):
            try:
                logger.info(f"Attempt {i}: {site_url}")
                response = requests.get(site_url, headers=headers)
                
                if response.status_code == 200:
                    site_data = response.json()
                    site_id = site_data.get('id')
                    site_display_name = site_data.get('displayName', 'Unknown')
                    
                    if site_id:
                        logger.info(f"✅ Found site: '{site_display_name}' (ID: {site_id})")
                        return site_id
                        
                elif response.status_code == 404:
                    logger.warning(f"❌ Site not found with this URL format")
                else:
                    logger.warning(f"❌ HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                logger.warning(f"❌ Exception: {e}")
        
        # If direct methods fail, try searching
        return self.search_for_site()
    
    def search_for_site(self) -> Optional[str]:
        """
        Search for the site by name
        """
        logger.info("Searching for site by name")
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            # Search for sites
            search_url = f"{self.graph_base_url}/sites?search=*"
            response = requests.get(search_url, headers=headers)
            
            if response.status_code == 200:
                search_data = response.json()
                sites = search_data.get('value', [])
                
                logger.info(f"Found {len(sites)} total sites. Looking for matches...")
                
                # Look for our site
                possible_matches = []
                for site in sites:
                    site_name = site.get('name', '').lower()
                    site_display_name = site.get('displayName', '').lower()
                    site_web_url = site.get('webUrl', '').lower()
                    
                    if (self.site_name.lower() in site_name or 
                        self.site_name.lower() in site_display_name or
                        self.site_name.lower() in site_web_url):
                        possible_matches.append(site)
                        
                if possible_matches:
                    # Use the first match
                    best_match = possible_matches[0]
                    site_id = best_match.get('id')
                    display_name = best_match.get('displayName', 'Unknown')
                    web_url = best_match.get('webUrl', 'Unknown')
                    
                    logger.info(f"✅ Found matching site: '{display_name}'")
                    logger.info(f"   URL: {web_url}")
                    logger.info(f"   ID: {site_id}")
                    
                    return site_id
                else:
                    logger.error(f"❌ No sites found matching '{self.site_name}'")
                    
                    # Show available sites for debugging
                    logger.info("Available sites:")
                    for i, site in enumerate(sites[:10], 1):  # Show first 10
                        logger.info(f"  {i}. '{site.get('displayName', 'Unknown')}' - {site.get('webUrl', 'Unknown')}")
                    
                    return None
            else:
                logger.error(f"Site search failed: HTTP {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Site search failed with exception: {e}")
            return None
            
    def get_drive_id(self, site_id: str) -> Optional[str]:
        """
        Get the default drive (document library) ID for the site
        """
        logger.info("Getting drive ID for document library")
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            drive_url = f"{self.graph_base_url}/sites/{site_id}/drive"
            response = requests.get(drive_url, headers=headers)
            response.raise_for_status()
            
            drive_data = response.json()
            drive_id = drive_data.get('id')
            drive_name = drive_data.get('name', 'Unknown')
            
            if drive_id:
                logger.info(f"✅ Found drive: '{drive_name}' (ID: {drive_id})")
                return drive_id
            else:
                logger.error("❌ Drive ID not found in response")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to get drive ID: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return None
            
    def upload_file(self, drive_id: str, file_path: str, sharepoint_filename: str) -> bool:
        """
        Upload a file to the SharePoint document library
        """
        logger.info(f"📤 Uploading {sharepoint_filename}")
        
        if not os.path.exists(file_path):
            logger.error(f"❌ File not found: {file_path}")
            return False
            
        try:
            # Read file content
            with open(file_path, 'rb') as file:
                file_content = file.read()
                
            file_size = len(file_content)
            logger.info(f"   File size: {file_size:,} bytes")
            
            # Prepare upload
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/octet-stream'
            }
            
            # Use simple upload for files under 4MB
            # For Shared Documents, upload to root of drive
            upload_url = f"{self.graph_base_url}/drives/{drive_id}/root:/{sharepoint_filename}:/content"
            
            response = requests.put(upload_url, headers=headers, data=file_content)
            
            if response.status_code in [200, 201]:
                upload_response = response.json()
                web_url = upload_response.get('webUrl', 'Unknown')
                
                logger.info(f"✅ Successfully uploaded {sharepoint_filename}")
                logger.info(f"   SharePoint URL: {web_url}")
                return True
            else:
                logger.error(f"❌ Upload failed: HTTP {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Upload failed with exception: {e}")
            return False
            
    def upload_csv_files(self) -> bool:
        """
        Main method to upload all CSV files
        """
        logger.info("🚀 Starting CSV upload process")
        
        # Define files to upload
        csv_files = [
            ('actuals.csv', 'actuals.csv'),
            ('predicts.csv', 'predicts.csv'), 
            ('forecasts.csv', 'forecasts.csv'),
            ('model_metrics.csv', 'model_metrics.csv')
        ]
        
        # Get upload directory
        project_root = os.path.dirname(os.path.dirname(__file__))
        upload_dir = os.path.join(project_root, 'data', 'upload')
        
        logger.info(f"📁 Upload directory: {upload_dir}")
        
        if not os.path.exists(upload_dir):
            logger.error(f"❌ Upload directory not found: {upload_dir}")
            return False
            
        # Check what files exist
        existing_files = [f for f in os.listdir(upload_dir) if f.endswith('.csv')]
        logger.info(f"📋 Found CSV files: {existing_files}")
        
        # Validate configuration
        if not self.validate_config():
            return False
            
        # Get access token
        if not self.get_access_token():
            logger.error("❌ Failed to get access token")
            return False
            
        # Find SharePoint site
        site_id = self.find_site_by_url()
        if not site_id:
            logger.error("❌ Could not find SharePoint site")
            return False
            
        # Get drive ID
        drive_id = self.get_drive_id(site_id)
        if not drive_id:
            logger.error("❌ Could not get drive ID")
            return False
            
        # Upload files
        upload_results = []
        successful_uploads = 0
        
        for local_filename, sharepoint_filename in csv_files:
            local_file_path = os.path.join(upload_dir, local_filename)
            
            if not os.path.exists(local_file_path):
                logger.warning(f"⚠️  File not found, skipping: {local_filename}")
                upload_results.append(False)
                continue
                
            file_size = os.path.getsize(local_file_path)
            if file_size == 0:
                logger.warning(f"⚠️  File is empty, skipping: {local_filename}")
                upload_results.append(False)
                continue
                
            # Upload file
            success = self.upload_file(drive_id, local_file_path, sharepoint_filename)
            upload_results.append(success)
            
            if success:
                successful_uploads += 1
                
        # Report results
        total_files = len([f for f in csv_files if os.path.exists(os.path.join(upload_dir, f[0]))])
        
        logger.info(f"📊 Upload Summary: {successful_uploads}/{total_files} files uploaded successfully")
        
        return successful_uploads > 0


def main():
    """
    Main execution function
    """
    logger.info("=" * 70)
    logger.info("🚀 SHAREPOINT CSV UPLOAD PROCESS STARTED")
    logger.info(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    try:
        uploader = SharePointUploader()
        success = uploader.upload_csv_files()
        
        if success:
            logger.info("=" * 70)
            logger.info("✅ SHAREPOINT CSV UPLOAD PROCESS COMPLETED SUCCESSFULLY")
            logger.info(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 70)
        else:
            raise Exception("SharePoint upload process failed")
            
    except Exception as e:
        logger.error("=" * 70)
        logger.error("❌ SHAREPOINT CSV UPLOAD PROCESS FAILED")
        logger.error(f"💥 Error: {str(e)}")
        logger.error(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.error("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
