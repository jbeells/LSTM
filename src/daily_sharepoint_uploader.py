#!/usr/bin/env python3
"""
SharePoint CSV Uploader - Daily Overwrite Version (FIXED)
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
        
        # Try NTLM authentication first (if available)
        try:
            from requests_ntlm import HttpNtlmAuth
            print("   🔑 Trying NTLM authentication...")
            
            auth = HttpNtlmAuth(self.username, self.password)
            
            response = self.session.post(
                digest_url,
                auth=auth,
                headers={
                    'Accept': 'application/json;odata=verbose',
                    'Content-Type': 'application/json;odata=verbose'
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                self.form_digest = data['d']['GetContextWebInformation']['FormDigestValue']
                print("   ✅ NTLM form digest obtained")
                return True
            else:
                print(f"   ⚠️  NTLM failed ({response.status_code}), trying basic auth...")
                
        except ImportError:
            print("   ⚠️  requests-ntlm not available, using basic auth...")
        except Exception as e:
            print(f"   ⚠️  NTLM error ({e}), trying basic auth...")
        
        # Fallback to basic authentication (ALWAYS TRY THIS)
        return self._try_basic_auth_digest()
    
    def _try_basic_auth_digest(self):
        """Basic authentication for form digest"""
        print("   🔑 Trying Basic authentication...")
        
        digest_url = f"{self.site_url}/_api/contextinfo"
        
        try:
            # Create basic auth header
            auth_string = f"{self.username}:{self.password}"
            auth_bytes = auth_string.encode('utf-8')
            auth_b64 = base64.b64encode(auth_bytes).decode('utf-8')
            
            response = self.session.post(
                digest_url,
                headers={
                    'Authorization': f'Basic {auth_b64}',
                    'Accept': 'application/json;odata=verbose',
                    'Content-Type': 'application/json;odata=verbose'
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                self.form_digest = data['d']['GetContextWebInformation']['FormDigestValue']
                print("   ✅ Basic auth form digest obtained")
                return True
            else:
                print(f"   ❌ Basic auth failed: {response.status_code}")
                if response.status_code == 401:
                    print("   🔍 Check username/password")
                elif response.status_code == 403:
                    print("   🔍 Check site permissions")
                elif response.status_code == 404:
                    print("   🔍 Check site URL")
                    
                print(f"   Response: {response.text[:200]}...")
                return False
                
        except Exception as e:
            print(f"   ❌ Basic auth error: {e}")
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
        
        # Use the same authentication method that worked for form digest
        return self._upload_with_auth(upload_url, file_content, filename)
    
    def _upload_with_auth(self, upload_url, file_content, filename):
        """Try upload with different auth methods"""
        
        headers = {
            'Accept': 'application/json;odata=verbose',
            'X-RequestDigest': self.form_digest,
            'Content-Type': 'application/octet-stream'
        }
        
        # Method 1: Try NTLM (if available)
        try:
            from requests_ntlm import HttpNtlmAuth
            auth = HttpNtlmAuth(self.username, self.password)
            
            response = self.session.post(
                upload_url,
                auth=auth,
                headers=headers,
                data=file_content,
                timeout=60
            )
            
            if response.status_code in [200, 201]:
                return self._handle_upload_success(response, file_content)
            else:
                print(f"   ⚠️  NTLM upload failed ({response.status_code}), trying basic auth...")
                
        except ImportError:
            pass  # NTLM not available
        except Exception as e:
            print(f"   ⚠️  NTLM upload error ({e}), trying basic auth...")
        
        # Method 2: Basic authentication (fallback)
        try:
            auth_string = f"{self.username}:{self.password}"
            auth_bytes = auth_string.encode('utf-8')
            auth_b64 = base64.b64encode(auth_bytes).decode('utf-8')
            
            headers['Authorization'] = f'Basic {auth_b64}'
            
            response = self.session.post(
                upload_url,
                headers=headers,
                data=file_content,
                timeout=60
            )
            
            if response.status_code in [200, 201]:
                return self._handle_upload_success(response, file_content)
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
    
    def _handle_upload_success(self, response, file_content):
        """Handle successful upload response"""
        print(f"   ✅ Upload successful - File overwritten")
        
        try:
            data = response.json()
            server_url = data.get('d', {}).get('ServerRelativeUrl', 'Unknown')
            file_size = len(file_content)
            print(f"   📍 URL: {server_url}")
            print(f"   📏 Size: {file_size} bytes")
            
            time_modified = data.get('d', {}).get('TimeLastModified', None)
            if time_modified:
                print(f"   🕒 Modified: {time_modified}")
                
        except Exception:
            print(f"   📏 Size: {len(file_content)} bytes")
        
        return True
    
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
