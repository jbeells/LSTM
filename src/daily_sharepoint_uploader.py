#!/usr/bin/env python3
"""
SharePoint CSV Uploader - Daily Overwrite Version
Files are overwritten each day with the same names
"""

import os
import requests
import json
import glob
from datetime import datetime
import base64

class SharePointBasicUploader:
    """SharePoint uploader using basic authentication"""
    
    def __init__(self):
        self.username = os.getenv('M365_USERNAME')
        self.password = os.getenv('M365_PASSWORD')
        self.site_url = os.getenv('SHAREPOINT_SITE_URL')
        self.folder_path = os.getenv('SHAREPOINT_FOLDER_PATH', 'Shared Documents')
        
        # Clean up site URL
        if self.site_url and self.site_url.endswith('/'):
            self.site_url = self.site_url[:-1]
            
        self.session = requests.Session()
        self.form_digest = None
        
    def get_form_digest(self):
        """Get form digest token for SharePoint operations"""
        print("🔐 Getting SharePoint form digest...")
        
        digest_url = f"{self.site_url}/_api/contextinfo"
        
        try:
            # Try NTLM authentication first
            from requests_ntlm import HttpNtlmAuth
            auth = HttpNtlmAuth(self.username, self.password)
            
            response = self.session.post(
                digest_url,
                auth=auth,
                headers={
                    'Accept': 'application/json;odata=verbose',
                    'Content-Type': 'application/json;odata=verbose'
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                self.form_digest = data['d']['GetContextWebInformation']['FormDigestValue']
                print("   ✅ NTLM form digest obtained")
                return True
            else:
                print(f"   ❌ NTLM form digest failed: {response.status_code}")
                return False
                
        except ImportError:
            print("   ⚠️  requests-ntlm not available, trying basic auth...")
            return self._try_basic_auth_digest()
        except Exception as e:
            print(f"   ❌ NTLM form digest error: {e}")
            return self._try_basic_auth_digest()
    
    def _try_basic_auth_digest(self):
        """Fallback to basic authentication"""
        digest_url = f"{self.site_url}/_api/contextinfo"
        
        try:
            # Basic authentication
            auth_string = f"{self.username}:{self.password}"
            auth_bytes = auth_string.encode('utf-8')
            auth_b64 = base64.b64encode(auth_bytes).decode('utf-8')
            
            response = self.session.post(
                digest_url,
                headers={
                    'Authorization': f'Basic {auth_b64}',
                    'Accept': 'application/json;odata=verbose',
                    'Content-Type': 'application/json;odata=verbose'
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                self.form_digest = data['d']['GetContextWebInformation']['FormDigestValue']
                print("   ✅ Basic auth form digest obtained")
                return True
            else:
                print(f"   ❌ Basic auth digest failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                return False
                
        except Exception as e:
            print(f"   ❌ Basic auth digest error: {e}")
            return False
    
    def upload_file(self, file_path, filename=None):
        """Upload a single file to SharePoint (overwrite existing)"""
        if not self.form_digest:
            if not self.get_form_digest():
                return False
        
        if not filename:
            filename = os.path.basename(file_path)
        
        print(f"📤 Uploading: {filename} (overwrite=true)")
        
        # Read file content
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
        except Exception as e:
            print(f"   ❌ Failed to read file: {e}")
            return False
        
        # Upload URL with overwrite=true
        upload_url = f"{self.site_url}/_api/web/getfolderbyserverrelativeurl('{self.folder_path}')/files/add(url='{filename}',overwrite=true)"
        
        try:
            headers = {
                'Accept': 'application/json;odata=verbose',
                'X-RequestDigest': self.form_digest,
                'Content-Type': 'application/octet-stream'
            }
            
            # Try NTLM authentication first
            try:
                from requests_ntlm import HttpNtlmAuth
                auth = HttpNtlmAuth(self.username, self.password)
                
                response = self.session.post(
                    upload_url,
                    auth=auth,
                    headers=headers,
                    data=file_content
                )
                
            except ImportError:
                # Fallback to basic authentication
                auth_string = f"{self.username}:{self.password}"
                auth_bytes = auth_string.encode('utf-8')
                auth_b64 = base64.b64encode(auth_bytes).decode('utf-8')
                headers['Authorization'] = f'Basic {auth_b64}'
                
                response = self.session.post(
                    upload_url,
                    headers=headers,
                    data=file_content
                )
            
            if response.status_code in [200, 201]:
                print(f"   ✅ Upload successful - File overwritten")
                
                # Parse response for details
                try:
                    data = response.json()
                    server_url = data.get('d', {}).get('ServerRelativeUrl', 'Unknown')
                    file_size = len(file_content)
                    print(f"   📍 URL: {server_url}")
                    print(f"   📏 Size: {file_size} bytes")
                    
                    # Get last modified time from response if available
                    time_modified = data.get('d', {}).get('TimeLastModified', None)
                    if time_modified:
                        print(f"   🕒 Modified: {time_modified}")
                        
                except Exception as parse_error:
                    print(f"   📏 Size: {len(file_content)} bytes")
                    print(f"   ⚠️  Could not parse response details: {parse_error}")
                
                return True
            else:
                print(f"   ❌ Upload failed: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', {}).get('value', 'Unknown error')
                    print(f"   Error: {error_msg}")
                except:
                    print(f"   Response: {response.text[:300]}...")
                return False
                
        except Exception as e:
            print(f"   ❌ Upload error: {e}")
            return False
    
    def upload_csv_files(self, directory="data/upload"):
        """Upload all CSV files from directory (daily overwrite)"""
        print(f"🔍 Looking for CSV files in {directory}...")
        
        csv_files = glob.glob(f"{directory}/*.csv")
        if not csv_files:
            print(f"   ❌ No CSV files found in {directory}")
            return False
        
        print(f"   📊 Found {len(csv_files)} CSV files")
        print("   📝 Files will overwrite existing versions in SharePoint")
        
        success_count = 0
        total_count = len(csv_files)
        
        for csv_file in csv_files:
            print(f"\n{'='*60}")
            if self.upload_file(csv_file):
                success_count += 1
            else:
                print(f"   ❌ Failed to upload {os.path.basename(csv_file)}")
        
        print(f"\n{'='*60}")
        print(f"📊 DAILY UPLOAD SUMMARY")
        print(f"Total files: {total_count}")
        print(f"Successful: {success_count}")
        print(f"Failed: {total_count - success_count}")
        
        if success_count == total_count:
            print("🎉 All files uploaded successfully!")
        elif success_count > 0:
            print("⚠️  Some files uploaded successfully, check logs above")
        else:
            print("❌ No files uploaded successfully")
            
        print(f"{'='*60}")
        
        return success_count > 0

def validate_environment():
    """Validate required environment variables"""
    required_vars = ['M365_USERNAME', 'M365_PASSWORD', 'SHAREPOINT_SITE_URL']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        return False
    
    return True

def main():
    """Main upload function"""
    print("🚀 DAILY SHAREPOINT CSV UPLOADER")
    print("=" * 50)
    print(f"⏰ Upload time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📝 Mode: Daily overwrite (no timestamps)")
    print()
    
    # Validate environment variables
    if not validate_environment():
        return False
    
    # Initialize uploader
    uploader = SharePointBasicUploader()
    
    print(f"🌐 Site: {uploader.site_url}")
    print(f"👤 User: {uploader.username}")
    print(f"📁 Folder: {uploader.folder_path}")
    print()
    
    # Upload CSV files
    try:
        if uploader.upload_csv_files():
            print("\n🎉 DAILY UPLOAD SUCCESSFUL!")
            print("📊 Fresh forecast data is now available in SharePoint")
            return True
        else:
            print("\n❌ DAILY UPLOAD FAILED")
            return False
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
