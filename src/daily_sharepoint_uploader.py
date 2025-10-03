#!/usr/bin/env python3
"""
Simple SharePoint Uploader - Basic Auth Only
Removes NTLM complexity and focuses on working basic authentication
"""

import os
import requests
import json
import glob
from datetime import datetime
import base64

class SimpleSharePointUploader:
    """Simple SharePoint uploader using only basic authentication"""
    
    def __init__(self):
        self.username = os.getenv('M365_USERNAME')
        self.password = os.getenv('M365_PASSWORD')
        self.site_url = os.getenv('SHAREPOINT_SITE_URL')
        self.folder_path = os.getenv('SHAREPOINT_FOLDER_PATH', 'Shared Documents')
        
        # Clean up site URL
        if self.site_url and self.site_url.endswith('/'):
            self.site_url = self.site_url[:-1]
        
        # Create auth header once
        auth_string = f"{self.username}:{self.password}"
        auth_bytes = auth_string.encode('utf-8')
        self.auth_header = base64.b64encode(auth_bytes).decode('utf-8')
        
        self.session = requests.Session()
        self.form_digest = None
        
        print(f"🌐 Site: {self.site_url}")
        print(f"👤 User: {self.username}")
        print(f"📁 Folder: {self.folder_path}")
        print()
        
    def get_form_digest(self):
        """Get form digest using basic authentication"""
        print("🔐 Getting SharePoint form digest (Basic Auth)...")
        
        digest_url = f"{self.site_url}/_api/contextinfo"
        
        headers = {
            'Authorization': f'Basic {self.auth_header}',
            'Accept': 'application/json;odata=verbose',
            'Content-Type': 'application/json;odata=verbose',
            'User-Agent': 'Python-SharePoint-Uploader/1.0'
        }
        
        try:
            print(f"   📡 POST {digest_url}")
            
            response = requests.post(
                digest_url, 
                headers=headers, 
                timeout=30,
                verify=True
            )
            
            print(f"   📊 Response: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    self.form_digest = data['d']['GetContextWebInformation']['FormDigestValue']
                    print(f"   ✅ Form digest obtained (length: {len(self.form_digest)})")
                    return True
                except Exception as parse_error:
                    print(f"   ❌ JSON parse error: {parse_error}")
                    print(f"   Raw response: {response.text[:300]}")
                    return False
                    
            elif response.status_code == 401:
                print("   ❌ Authentication failed - Check username/password")
                print("   🔍 Verify M365_USERNAME and M365_PASSWORD secrets")
                return False
                
            elif response.status_code == 403:
                print("   ❌ Access forbidden - Check permissions")
                print("   🔍 User needs 'Contribute' or 'Edit' permissions on the site")
                print(f"   🔍 Site URL: {self.site_url}")
                return False
                
            elif response.status_code == 404:
                print("   ❌ Site not found")
                print("   🔍 Check SHAREPOINT_SITE_URL - should be full site URL")
                print(f"   🔍 Current: {self.site_url}")
                return False
                
            else:
                print(f"   ❌ Unexpected response: {response.status_code}")
                print(f"   Headers: {dict(response.headers)}")
                print(f"   Body: {response.text[:500]}")
                return False
                
        except requests.exceptions.Timeout:
            print("   ❌ Request timeout - Site may be slow or unreachable")
            return False
        except requests.exceptions.ConnectionError:
            print("   ❌ Connection error - Check site URL and internet connection")
            return False
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            return False
    
    def upload_file(self, file_path):
        """Upload a single file"""
        filename = os.path.basename(file_path)
        print(f"📤 Uploading: {filename}")
        
        # Ensure we have form digest
        if not self.form_digest:
            if not self.get_form_digest():
                return False
        
        # Read file
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
            print(f"   📏 File size: {len(file_content)} bytes")
        except Exception as e:
            print(f"   ❌ Failed to read file: {e}")
            return False
        
        # Upload
        upload_url = f"{self.site_url}/_api/web/getfolderbyserverrelativeurl('{self.folder_path}')/files/add(url='{filename}',overwrite=true)"
        
        headers = {
            'Authorization': f'Basic {self.auth_header}',
            'Accept': 'application/json;odata=verbose',
            'X-RequestDigest': self.form_digest,
            'Content-Type': 'application/octet-stream'
        }
        
        try:
            print(f"   📡 Uploading to SharePoint...")
            
            response = requests.post(
                upload_url,
                headers=headers,
                data=file_content,
                timeout=120  # 2 minutes for large files
            )
            
            if response.status_code in [200, 201]:
                print(f"   ✅ Upload successful!")
                try:
                    data = response.json()
                    server_url = data.get('d', {}).get('ServerRelativeUrl', '')
                    if server_url:
                        print(f"   📍 SharePoint URL: {server_url}")
                except:
                    pass
                return True
            else:
                print(f"   ❌ Upload failed: {response.status_code}")
                if response.status_code == 404:
                    print(f"   🔍 Check folder path: {self.folder_path}")
                elif response.status_code == 403:
                    print("   🔍 Check write permissions to folder")
                print(f"   Response: {response.text[:300]}")
                return False
                
        except Exception as e:
            print(f"   ❌ Upload error: {e}")
            return False
    
    def upload_csv_files(self):
        """Upload all CSV files"""
        csv_directory = "data/upload"
        print(f"🔍 Looking for CSV files in {csv_directory}...")
        
        csv_files = glob.glob(f"{csv_directory}/*.csv")
        if not csv_files:
            print(f"   ❌ No CSV files found")
            return False
        
        print(f"   📊 Found {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            print(f"      - {os.path.basename(csv_file)}")
        print()
        
        success_count = 0
        
        for csv_file in csv_files:
            print("=" * 50)
            if self.upload_file(csv_file):
                success_count += 1
                print(f"   🎉 {os.path.basename(csv_file)} uploaded successfully")
            else:
                print(f"   💥 {os.path.basename(csv_file)} upload failed")
            print()
        
        print("=" * 50)
        print(f"📊 UPLOAD SUMMARY:")
        print(f"   Total files: {len(csv_files)}")
        print(f"   Successful: {success_count}")
        print(f"   Failed: {len(csv_files) - success_count}")
        
        return success_count > 0

def main():
    """Main function"""
    print("🚀 SIMPLE SHAREPOINT CSV UPLOADER")
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
    
    # Upload
    uploader = SimpleSharePointUploader()
    
    if uploader.upload_csv_files():
        print("🎉 ALL UPLOADS SUCCESSFUL!")
        return True
    else:
        print("❌ SOME UPLOADS FAILED")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
