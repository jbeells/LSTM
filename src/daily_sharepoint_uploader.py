#!/usr/bin/env python3
"""
SharePoint CSV Uploader - Working Scope Version
==============================================

Uses the SharePoint-specific scope that was confirmed working in diagnostics.

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
    SharePoint uploader using SharePoint-specific token scope
    """
    
    def __init__(self):
        """Initialize with configuration"""
        self.tenant_id = os.getenv('SHAREPOINT_TENANT_ID')
        self.client_id = os.getenv('SHAREPOINT_CLIENT_ID') 
        self.client_secret = os.getenv('SHAREPOINT_CLIENT_SECRET')
        
        self.site_url = os.getenv('SHAREPOINT_SITE_URL', 'https://jeanalytics.sharepoint.com/sites/LSTM')
        self.folder_path = os.getenv('SHAREPOINT_FOLDER_PATH', 'Shared Documents')
        
        # Extract components
        if 'sharepoint.com' in self.site_url:
            url_parts = self.site_url.replace('https://', '').split('/')
            self.tenant_name = url_parts[0].split('.')[0]
            self.hostname = url_parts[0]
            self.site_name = url_parts[-1] if len(url_parts) > 2 else 'root'
        else:
            raise ValueError(f"Invalid SharePoint site URL: {self.site_url}")
        
        # Use SharePoint-specific endpoints
        self.sharepoint_base_url = self.site_url
        self.token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        self.access_token = None
        
        logger.info(f"🚀 SharePoint uploader initialized")
        logger.info(f"   Site: {self.site_url}")
        logger.info(f"   Using SharePoint-specific authentication")
        
    def validate_config(self) -> bool:
        """Validate configuration"""
        required_vars = {
            'SHAREPOINT_TENANT_ID': self.tenant_id,
            'SHAREPOINT_CLIENT_ID': self.client_id, 
            'SHAREPOINT_CLIENT_SECRET': self.client_secret
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            logger.error(f"❌ Missing environment variables: {missing_vars}")
            return False
            
        logger.info("✅ Configuration validated")
        return True
        
    def get_sharepoint_token(self) -> bool:
        """
        Get SharePoint-specific access token using the working scope
        """
        logger.info("🔑 Getting SharePoint-specific access token")
        
        # Use the scope that worked in diagnostics
        token_data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': f'https://{self.hostname}/.default',
            'grant_type': 'client_credentials'
        }
        
        try:
            response = requests.post(self.token_url, data=token_data)
            response.raise_for_status()
            
            token_response = response.json()
            self.access_token = token_response.get('access_token')
            
            if self.access_token:
                token_type = token_response.get('token_type', 'Bearer')
                expires_in = token_response.get('expires_in', 'unknown')
                
                logger.info(f"✅ Got SharePoint token (expires in {expires_in}s)")
                return True
            else:
                logger.error("❌ No access token in response")
                return False
                
        except Exception as e:
            logger.error(f"❌ Token request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return False
    
    def test_sharepoint_rest_api(self) -> bool:
        """
        Test SharePoint REST API access
        """
        logger.info("🧪 Testing SharePoint REST API access")
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json;odata=verbose',
            'Content-Type': 'application/json'
        }
        
        # Test basic site info
        test_url = f"{self.sharepoint_base_url}/_api/web"
        
        try:
            response = requests.get(test_url, headers=headers)
            
            if response.status_code == 200:
                site_data = response.json()
                site_title = site_data.get('d', {}).get('Title', 'Unknown')
                logger.info(f"✅ SharePoint REST API working - Site: '{site_title}'")
                return True
            else:
                logger.error(f"❌ REST API test failed: HTTP {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ REST API test exception: {e}")
            return False
    
    def get_documents_library_info(self) -> Optional[Dict]:
        """
        Get information about the Documents library
        """
        logger.info("📚 Getting Documents library information")
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json;odata=verbose',
            'Content-Type': 'application/json'
        }
        
        try:
            # Get the Documents library (default document library)
            library_url = f"{self.sharepoint_base_url}/_api/web/lists/getbytitle('Documents')"
            response = requests.get(library_url, headers=headers)
            
            if response.status_code == 200:
                library_data = response.json()
                library_info = library_data.get('d', {})
                
                title = library_info.get('Title', 'Unknown')
                item_count = library_info.get('ItemCount', 'Unknown')
                
                logger.info(f"✅ Found Documents library: '{title}' ({item_count} items)")
                return library_info
            else:
                logger.warning(f"⚠️  Documents library access failed: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️  Documents library check failed: {e}")
            return None
    
    def upload_file_to_sharepoint(self, file_path: str, sharepoint_filename: str) -> bool:
        """
        Upload file using SharePoint REST API
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
            logger.info(f"   📁 File size: {file_size:,} bytes")
            
            # Upload to Documents library using REST API
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json;odata=verbose',
                'Content-Type': 'application/octet-stream'
            }
            
            # SharePoint REST API endpoint for file upload
            upload_url = f"{self.sharepoint_base_url}/_api/web/lists/getbytitle('Documents')/RootFolder/Files/add(url='{sharepoint_filename}',overwrite=true)"
            
            response = requests.post(upload_url, headers=headers, data=file_content)
            
            if response.status_code in [200, 201]:
                logger.info(f"✅ Successfully uploaded {sharepoint_filename}")
                
                # Try to get the file URL from response
                try:
                    upload_response = response.json()
                    server_relative_url = upload_response.get('d', {}).get('ServerRelativeUrl', '')
                    if server_relative_url:
                        file_url = f"{self.sharepoint_base_url.rstrip('/')}{server_relative_url}"
                        logger.info(f"   🔗 File URL: {file_url}")
                except:
                    logger.info(f"   📍 File uploaded to Documents library")
                
                return True
            else:
                logger.error(f"❌ Upload failed: HTTP {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Upload exception: {e}")
            return False
    
    def upload_csv_files(self) -> bool:
        """
        Upload all CSV files to SharePoint
        """
        logger.info("🚀 Starting CSV file upload process")
        
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
        
        logger.info(f"📁 Looking in: {upload_dir}")
        
        if not os.path.exists(upload_dir):
            logger.error(f"❌ Upload directory not found: {upload_dir}")
            return False
        
        # Check files
        existing_files = []
        for local_filename, sharepoint_filename in csv_files:
            file_path = os.path.join(upload_dir, local_filename)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                existing_files.append((local_filename, sharepoint_filename, file_path))
            else:
                logger.warning(f"⚠️  Skipping {local_filename} (not found or empty)")
        
        if not existing_files:
            logger.error("❌ No CSV files found to upload")
            return False
            
        logger.info(f"📋 Found {len(existing_files)} files to upload")
        
        # Validate configuration
        if not self.validate_config():
            return False
        
        # Get SharePoint token
        if not self.get_sharepoint_token():
            return False
        
        # Test SharePoint access
        if not self.test_sharepoint_rest_api():
            logger.error("❌ SharePoint REST API access failed")
            return False
        
        # Get Documents library info
        self.get_documents_library_info()
        
        # Upload files
        upload_results = []
        successful_uploads = 0
        
        for local_filename, sharepoint_filename, file_path in existing_files:
            logger.info(f"📤 Processing: {local_filename} → {sharepoint_filename}")
            
            success = self.upload_file_to_sharepoint(file_path, sharepoint_filename)
            upload_results.append(success)
            
            if success:
                successful_uploads += 1
            
            # Small delay between uploads
            import time
            time.sleep(1)
        
        # Report results
        total_files = len(existing_files)
        logger.info(f"📊 Upload Summary: {successful_uploads}/{total_files} files uploaded successfully")
        
        if successful_uploads == total_files:
            logger.info("✅ All files uploaded successfully!")
            return True
        elif successful_uploads > 0:
            logger.warning(f"⚠️  Partial success: {successful_uploads}/{total_files} files uploaded")
            return True
        else:
            logger.error("❌ No files were uploaded")
            return False


def main():
    """
    Main execution function
    """
    logger.info("=" * 80)
    logger.info("🚀 SHAREPOINT CSV UPLOAD - WORKING SCOPE VERSION")
    logger.info(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    try:
        uploader = SharePointUploader()
        success = uploader.upload_csv_files()
        
        if success:
            logger.info("=" * 80)
            logger.info("✅ SHAREPOINT UPLOAD COMPLETED SUCCESSFULLY")
            logger.info(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 80)
        else:
            raise Exception("Upload process failed")
            
    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ SHAREPOINT UPLOAD FAILED")
        logger.error(f"💥 Error: {str(e)}")
        logger.error(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
