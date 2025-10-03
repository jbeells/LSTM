#!/usr/bin/env python3
"""
SharePoint CSV Uploader - Working Version
Fixed authentication fallback logic
"""

import os
import requests
import json
import glob
from datetime import datetime
import base64

class SharePointBasicUploader:
    """SharePoint uploader with proper authentication fallback"""
    
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
        """Get form digest with proper fallback logic"""
        print("🔐 Getting SharePoint form digest...")
        
        digest_url = f"{self.site_url}/_api/contextinfo"
        print(f"   📡 Requesting: {digest_url}")
        
        # Try NTLM first (if available)
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
            
            print(f"   📊 NTLM Response: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    self.form_digest = data['d']['GetContextWebInformation']['FormDigestValue']
                    print("   ✅ NTLM form digest obtained")
                    return True
                except Exception as e:
                    print(f"   ⚠️  NTLM JSON parse error: {e}")
            else:
                print(f"   ⚠️  NTLM failed with {response.status_code}, trying basic auth...")
                
        except ImportError:
            print("   ⚠️  requests-ntlm not available, trying basic auth...")
        except Exception as e:
            print(f"   ⚠️  NTLM error: {e}, trying basic auth...")
        
        # ALWAYS try basic auth as fallback
        return self._try_basic_auth_digest()
    
    def _try_basic_auth_digest(self):
        """Basic authentication fallback"""
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
                    'Content-Type': 'application/json;odata=verbose',
                    'User-Agent': 'Python-SharePoint-Uploader/1.0'
                },
                timeout=30
            )
            
            print(f"   📊 Basic Auth Response: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    self.form_digest = data['d']['GetContextWebInformation']['FormDigestValue']
                    print("   ✅ Basic auth form digest obtained")
                    return True
                except Exception as e:
                    print(f"   ❌ Basic auth JSON parse error: {e}")
                    print(f"   Raw response: {response.text[:300]}")
                    return False
                    
            elif response.status_code == 401:
                print("   ❌ Basic auth failed: Invalid credentials")
                print("   🔍 Check M365_USERNAME and M365_PASSWORD secrets")
                
            elif response.status_code == 403:
                print("   ❌ Basic auth failed: Access forbidden")
                print("   🔍 Check user permissions on SharePoint site")
                print(f"   🔍 Site: {self.site_url}")
                
            elif response.status_code == 404:
                print("   ❌ Basic auth failed: Site not found")
                print("   🔍 Check SHAREPOINT_SITE_URL format")
                print(f"   🔍 Current: {self.site_url}")
                
            else:
                print(f"   ❌ Basic auth failed: HTTP {response.status_code}")
                
            print(f"   🔍 Response: {response.text[:200]}...")
            return False
                
        except Exception as e:
            print(f"   ❌ Basic auth error: {e}")
            return False
    
    def upload_file(self, file_path, filename=None):
        """Upload file with authentication"""
        if not self.form_digest:
            if not self.get_form_digest():
                return False
        
        if not filename:
            filename = os.path.basename(file_path)
        
        print(f"📤 Uploading: {filename}")
        
        # Read file
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
            print(f"   📏 Size: {len(file_content)} bytes")
        except Exception as e:
            print(f"   ❌ Read error: {e}")
            return False
        
        # Upload with same auth that worked for form digest
        upload_url = f"{self.site_url}/_api/web/getfolderbyserverrelativeurl('{self.folder_path}')/files/add(url='{filename}',overwrite=true)"
        
        # Try basic auth (most likely to work)
        return self._upload_basic_auth(upload_url, file_content, filename)
    
    def _upload_basic_auth(self, upload_url, file_content, filename):
        """Upload using basic authentication"""
        auth_string = f"{self.username}:{self.password}"
        auth_bytes = auth_string.encode('utf-8')
        auth_b64 = base64.b64encode(auth_bytes).decode('utf-8')
        
        headers = {
            'Authorization': f'Basic {auth_b64}',
            'Accept': 'application/json;odata=verbose',
            'X-RequestDigest': self.form_digest,
            'Content-Type': 'application/octet-stream'
        }
        
        try:
            print(f"   📡 Uploading to SharePoint...")
            
            response = self.session.post(
                upload_url,
                headers=headers,
                data=file_content,
                timeout=120
            )
            
            print(f"   📊 Upload Response: {response.status_code}")
            
            if response.status_code in [200, 201]:
                print(f"   ✅ Upload successful!")
                return True
            else:
                print(f"   ❌ Upload failed: {response.status_code}")
                if response.status_code == 404:
                    print(f"   🔍 Check folder path: {self.folder_path}")
                elif response.status_code == 403:
                    print("   🔍 Check write permissions")
                print(f"   Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            print(f"   ❌ Upload error: {e}")
            return False
    
    def upload_csv_files(self, directory="data/upload"):
        """Upload all CSV files"""
        print(f"🔍 Looking for CSV files in {directory}...")
        
        csv_files = glob.glob(f"{directory}/*.csv")
        if not csv_files:
            print(f"   ❌ No CSV files found")
            return False
        
        print(f"   📊 Found {len(csv_files)} CSV files")
        for f in csv_files:
            print(f"      - {os.path.basename(f)}")
        print()
        
        success_count = 0
        
        for csv_file in csv_files:
            print(f"{'='*60}")
            if self.upload_file(csv_file):
                success_count += 1
            print()
        
        print(f"{'='*60}")
        print(f"📊 SUMMARY: {success_count}/{len(csv_files)} successful")
        
        return success_count > 0

def main():
    """Main function"""
    print("🚀 SHAREPOINT CSV UPLOADER (WORKING VERSION)")
    print("=" * 50)
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check environment
    required_vars = ['M365_USERNAME', 'M365_PASSWORD', 'SHAREPOINT_SITE_URL']
    missing = [v for v in required_vars if not os.getenv(v)]
    
    if missing:
        print("❌ Missing environment variables:")
        for var in missing:
            print(f"   - {var}")
        return False
    
    # Initialize
    uploader = SharePointBasicUploader()
    
    print(f"🌐 Site: {uploader.site_url}")
    print(f"👤 User: {uploader.username}")
    print(f"📁 Folder: {uploader.folder_path}")
    print()
    
    # Upload
    if uploader.upload_csv_files():
        print("🎉 UPLOADS SUCCESSFUL!")
        return True
    else:
        print("❌ UPLOADS FAILED")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
