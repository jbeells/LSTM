#!/usr/bin/env python3
"""
SharePoint CSV Uploader - Diagnostic Version
===========================================

This version includes comprehensive diagnostics to identify permission and configuration issues.

Author: AI Assistant
Date: 2024-10-03
"""

import os
import sys
import logging
import requests
import json
import base64
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

class SharePointDiagnostic:
    """
    SharePoint diagnostic and uploader with comprehensive troubleshooting
    """
    
    def __init__(self):
        """Initialize with configuration from environment variables"""
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
        
        self.graph_base_url = "https://graph.microsoft.com/v1.0"
        self.token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        self.access_token = None
        
        logger.info(f"🔧 Diagnostic mode initialized")
        logger.info(f"   Site: {self.site_url}")
        logger.info(f"   Tenant: {self.tenant_name}")
        
    def validate_config(self) -> bool:
        """Validate configuration"""
        required_vars = {
            'SHAREPOINT_TENANT_ID': self.tenant_id,
            'SHAREPOINT_CLIENT_ID': self.client_id, 
            'SHAREPOINT_CLIENT_SECRET': self.client_secret
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            logger.error(f"❌ Missing required environment variables: {missing_vars}")
            return False
            
        logger.info("✅ All required configuration variables are present")
        return True
        
    def get_access_token(self) -> bool:
        """Get access token and analyze it"""
        logger.info("🔑 Requesting Microsoft Graph access token")
        
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
                logger.info("✅ Successfully obtained access token")
                
                # Analyze the token
                self.analyze_token()
                return True
            else:
                logger.error("❌ Access token not found in response")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to obtain access token: {e}")
            return False
    
    def analyze_token(self):
        """Analyze the JWT token to understand its contents"""
        logger.info("🔍 Analyzing access token")
        
        try:
            # JWT tokens have 3 parts separated by dots
            token_parts = self.access_token.split('.')
            
            if len(token_parts) >= 2:
                # Decode the payload (second part)
                # Add padding if needed
                payload = token_parts[1]
                padding = len(payload) % 4
                if padding:
                    payload += '=' * (4 - padding)
                
                decoded = base64.b64decode(payload)
                token_data = json.loads(decoded)
                
                logger.info("📋 Token Analysis:")
                logger.info(f"   Issuer: {token_data.get('iss', 'Unknown')}")
                logger.info(f"   Audience: {token_data.get('aud', 'Unknown')}")
                logger.info(f"   App ID: {token_data.get('appid', 'Unknown')}")
                logger.info(f"   Tenant ID: {token_data.get('tid', 'Unknown')}")
                
                # Check permissions/roles
                roles = token_data.get('roles', [])
                if roles:
                    logger.info(f"   Granted Roles: {roles}")
                else:
                    logger.warning("   ⚠️  No roles found in token")
                
                # Check scopes
                scp = token_data.get('scp', '')
                if scp:
                    logger.info(f"   Scopes: {scp}")
                else:
                    logger.warning("   ⚠️  No scopes found in token")
                    
        except Exception as e:
            logger.warning(f"⚠️  Could not analyze token: {e}")
    
    def test_basic_graph_access(self) -> bool:
        """Test basic Microsoft Graph API access"""
        logger.info("🧪 Testing basic Graph API access")
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        # Test basic Graph access
        test_endpoints = [
            ("/me", "User profile (should fail for app-only)"),
            ("/applications", "Applications (requires Directory permissions)"),
            ("/organization", "Organization info"),
            ("/sites", "Sites (basic access)")
        ]
        
        for endpoint, description in test_endpoints:
            try:
                url = f"https://graph.microsoft.com/v1.0{endpoint}"
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    logger.info(f"   ✅ {description}: SUCCESS")
                elif response.status_code == 403:
                    logger.info(f"   🔒 {description}: FORBIDDEN (no permission)")
                elif response.status_code == 401:
                    logger.info(f"   ❌ {description}: UNAUTHORIZED")
                else:
                    logger.info(f"   ❓ {description}: HTTP {response.status_code}")
                    
            except Exception as e:
                logger.info(f"   💥 {description}: EXCEPTION - {e}")
    
    def test_sharepoint_permissions(self) -> bool:
        """Test different SharePoint permission levels"""
        logger.info("🔐 Testing SharePoint-specific permissions")
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        # Test different SharePoint endpoints to understand permissions
        test_cases = [
            # Basic sites access
            {
                'url': f"{self.graph_base_url}/sites",
                'description': 'List all sites',
                'expected': 'Should work with Sites.Read.All or Sites.ReadWrite.All'
            },
            # Specific tenant root site
            {
                'url': f"{self.graph_base_url}/sites/{self.hostname}",
                'description': 'Access tenant root site',
                'expected': 'Should work if tenant allows app access'
            },
            # Search for sites
            {
                'url': f"{self.graph_base_url}/sites?search=*",
                'description': 'Search sites',
                'expected': 'Should work with basic Sites permissions'
            }
        ]
        
        any_success = False
        
        for test_case in test_cases:
            try:
                logger.info(f"   Testing: {test_case['description']}")
                logger.info(f"   URL: {test_case['url']}")
                
                response = requests.get(test_case['url'], headers=headers)
                
                if response.status_code == 200:
                    logger.info(f"   ✅ SUCCESS: {test_case['description']}")
                    
                    # Try to get some data
                    data = response.json()
                    if 'value' in data:
                        count = len(data['value'])
                        logger.info(f"      Found {count} items")
                        any_success = True
                    else:
                        logger.info(f"      Single item response")
                        any_success = True
                        
                elif response.status_code == 401:
                    error_data = response.json() if response.text else {}
                    error_code = error_data.get('error', {}).get('code', 'unknown')
                    logger.error(f"   ❌ UNAUTHORIZED: {test_case['description']}")
                    logger.error(f"      Error code: {error_code}")
                    logger.error(f"      Expected: {test_case['expected']}")
                    
                elif response.status_code == 403:
                    logger.error(f"   🔒 FORBIDDEN: {test_case['description']}")
                    logger.error(f"      Expected: {test_case['expected']}")
                    
                else:
                    logger.warning(f"   ❓ HTTP {response.status_code}: {test_case['description']}")
                    
            except Exception as e:
                logger.error(f"   💥 EXCEPTION: {test_case['description']} - {e}")
        
        return any_success
    
    def try_alternative_upload_methods(self) -> bool:
        """Try alternative upload methods if Graph API fails"""
        logger.info("🔄 Trying alternative upload methods")
        
        csv_files = ['actuals.csv', 'predicts.csv', 'forecasts.csv', 'model_metrics.csv']
        project_root = os.path.dirname(os.path.dirname(__file__))
        upload_dir = os.path.join(project_root, 'data', 'upload')
        
        # Check if we have files to upload
        existing_files = []
        for filename in csv_files:
            file_path = os.path.join(upload_dir, filename)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                existing_files.append((filename, file_path))
        
        if not existing_files:
            logger.warning("⚠️  No CSV files found to upload")
            return False
            
        logger.info(f"📋 Found {len(existing_files)} files to upload: {[f[0] for f in existing_files]}")
        
        # Method 1: Try direct site access with known site ID format
        logger.info("🔍 Method 1: Direct site ID approach")
        success = self.try_direct_site_access(existing_files)
        if success:
            return True
            
        # Method 2: Try with different authentication scopes
        logger.info("🔍 Method 2: Alternative token scopes")
        success = self.try_alternative_scopes(existing_files)
        if success:
            return True
            
        logger.error("❌ All alternative upload methods failed")
        return False
    
    def try_direct_site_access(self, files_to_upload) -> bool:
        """Try accessing site with constructed site ID"""
        logger.info("   Trying direct site access with constructed ID")
        
        # SharePoint site IDs typically follow a pattern
        # Let's try a few common formats
        site_id_patterns = [
            f"{self.hostname},{self.tenant_id},{self.site_name}",
            f"{self.hostname}:/{self.site_name}",
        ]
        
        for pattern in site_id_patterns:
            logger.info(f"   Trying site ID pattern: {pattern}")
            # This would require more complex logic to test
            # For now, just log the attempt
        
        return False
    
    def try_alternative_scopes(self, files_to_upload) -> bool:
        """Try getting token with different scopes"""
        logger.info("   Trying alternative token scopes")
        
        alternative_scopes = [
            f"https://{self.hostname}/AllSites.Write",
            f"https://{self.hostname}/.default",
            "https://graph.microsoft.com/Sites.ReadWrite.All"
        ]
        
        for scope in alternative_scopes:
            logger.info(f"   Trying scope: {scope}")
            
            token_data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'scope': scope,
                'grant_type': 'client_credentials'
            }
            
            try:
                response = requests.post(self.token_url, data=token_data)
                if response.status_code == 200:
                    token_response = response.json()
                    alt_token = token_response.get('access_token')
                    
                    if alt_token:
                        logger.info(f"   ✅ Got token with scope: {scope}")
                        # Here you could try using this token for site access
                        # For now, just log success
                    else:
                        logger.info(f"   ❌ No token with scope: {scope}")
                else:
                    logger.info(f"   ❌ Failed to get token with scope: {scope}")
                    
            except Exception as e:
                logger.info(f"   💥 Exception with scope {scope}: {e}")
        
        return False
    
    def run_diagnostics(self) -> bool:
        """Run comprehensive diagnostics"""
        logger.info("🔧 Starting SharePoint diagnostics")
        
        # Step 1: Validate config
        if not self.validate_config():
            return False
        
        # Step 2: Get access token
        if not self.get_access_token():
            return False
        
        # Step 3: Test basic Graph access
        self.test_basic_graph_access()
        
        # Step 4: Test SharePoint permissions
        sharepoint_access = self.test_sharepoint_permissions()
        
        if sharepoint_access:
            logger.info("✅ SharePoint access confirmed - proceeding with upload")
            return True
        else:
            logger.error("❌ No SharePoint access detected")
            
            # Step 5: Try alternative methods
            return self.try_alternative_upload_methods()


def main():
    """
    Main diagnostic function
    """
    logger.info("=" * 80)
    logger.info("🔧 SHAREPOINT DIAGNOSTIC AND UPLOAD PROCESS")
    logger.info(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    try:
        diagnostic = SharePointDiagnostic()
        success = diagnostic.run_diagnostics()
        
        if success:
            logger.info("=" * 80)
            logger.info("✅ DIAGNOSTIC COMPLETED - READY FOR UPLOAD")
            logger.info(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 80)
        else:
            logger.error("=" * 80)
            logger.error("❌ DIAGNOSTIC IDENTIFIED CONFIGURATION ISSUES")
            logger.error("💡 RECOMMENDATIONS:")
            logger.error("   1. Verify App Registration has 'Sites.ReadWrite.All' permission")
            logger.error("   2. Ensure admin consent is granted (green checkmark in Azure)")
            logger.error("   3. Check if tenant allows app-only access to SharePoint")
            logger.error("   4. Verify the SharePoint site URL is correct")
            logger.error("   5. Consider asking SharePoint admin to add app to site collection")
            logger.error("=" * 80)
            
    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ DIAGNOSTIC PROCESS FAILED")
        logger.error(f"💥 Error: {str(e)}")
        logger.error(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
