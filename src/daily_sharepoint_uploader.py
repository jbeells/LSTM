#!/usr/bin/env python3
"""
Final test to verify SharePoint permissions are working
Since the app is registered in Site Collection App Permissions
"""

import os
import requests
import json
from datetime import datetime

def test_sharepoint_final():
    """Final test with registered app"""
    
    print("🚀 FINAL SHAREPOINT PERMISSIONS TEST")
    print("=" * 60)
    print(f"⏰ Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get environment variables
    tenant_id = os.getenv('SHAREPOINT_TENANT_ID')
    client_id = os.getenv('SHAREPOINT_CLIENT_ID')
    client_secret = os.getenv('SHAREPOINT_CLIENT_SECRET')
    site_url = os.getenv('SHAREPOINT_SITE_URL', 'https://jeanalytics.sharepoint.com/sites/LSTM')
    
    if not all([tenant_id, client_id, client_secret]):
        print("❌ Missing environment variables")
        return False
    
    print(f"📋 Client ID: {client_id}")
    print(f"📋 Tenant ID: {tenant_id}")
    print(f"📋 Site URL: {site_url}")
    print()
    
    # Test multiple authentication methods
    auth_methods = [
        {
            'name': 'Microsoft Graph API',
            'scope': 'https://graph.microsoft.com/.default',
            'test_func': test_graph_api
        },
        {
            'name': 'SharePoint Specific',
            'scope': 'https://jeanalytics.sharepoint.com/.default',
            'test_func': test_sharepoint_rest
        }
    ]
    
    for method in auth_methods:
        print(f"🔑 Testing: {method['name']}")
        print("-" * 40)
        
        # Get token
        token = get_token(tenant_id, client_id, client_secret, method['scope'])
        if not token:
            print(f"❌ Failed to get token for {method['name']}")
            continue
            
        print(f"✅ Got token for {method['name']}")
        
        # Test the method
        if method['test_func'](token, site_url):
            print(f"✅ {method['name']} - SUCCESS!")
            print()
            return True
        else:
            print(f"❌ {method['name']} - Failed")
            print()
    
    print("❌ All authentication methods failed")
    return False

def get_token(tenant_id, client_id, client_secret, scope):
    """Get access token for given scope"""
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    
    token_data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'scope': scope,
        'grant_type': 'client_credentials'
    }
    
    try:
        response = requests.post(token_url, data=token_data)
        if response.status_code == 200:
            return response.json().get('access_token')
        else:
            print(f"   Token request failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"   Token exception: {e}")
        return None

def test_graph_api(token, site_url):
    """Test Microsoft Graph API access"""
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    # Try different Graph API formats
    test_urls = [
        "https://graph.microsoft.com/v1.0/sites/jeanalytics.sharepoint.com:/sites/LSTM",
        "https://graph.microsoft.com/v1.0/sites/jeanalytics.sharepoint.com/sites/LSTM"
    ]
    
    for i, test_url in enumerate(test_urls, 1):
        print(f"   Attempt {i}: {test_url}")
        try:
            response = requests.get(test_url, headers=headers)
            if response.status_code == 200:
                site_data = response.json()
                site_name = site_data.get('displayName', 'Unknown')
                site_id = site_data.get('id', 'Unknown')
                
                print(f"   ✅ Site found: {site_name}")
                print(f"   ✅ Site ID: {site_id}")
                
                # Test drive access
                return test_drive_access(token, site_id)
                
            else:
                print(f"   ❌ HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    return False

def test_drive_access(token, site_id):
    """Test drive access and file operations"""
    print("   📁 Testing drive access...")
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive"
    
    try:
        response = requests.get(drive_url, headers=headers)
        if response.status_code == 200:
            drive_data = response.json()
            drive_name = drive_data.get('name', 'Unknown')
            drive_id = drive_data.get('id', 'Unknown')
            
            print(f"   ✅ Drive access: {drive_name}")
            print(f"   ✅ Drive ID: {drive_id}")
            
            # Test root folder access
            return test_root_folder(token, drive_id)
        else:
            print(f"   ❌ Drive access failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Drive access exception: {e}")
        return False

def test_root_folder(token, drive_id):
    """Test root folder access"""
    print("   📂 Testing root folder access...")
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    root_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/children"
    
    try:
        response = requests.get(root_url, headers=headers)
        if response.status_code == 200:
            files_data = response.json()
            file_count = len(files_data.get('value', []))
            
            print(f"   ✅ Root folder access successful")
            print(f"   ✅ Found {file_count} items in root folder")
            print(f"   🎯 READY FOR FILE UPLOAD!")
            return True
        else:
            print(f"   ❌ Root folder access failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Root folder exception: {e}")
        return False

def test_sharepoint_rest(token, site_url):
    """Test SharePoint REST API"""
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json;odata=verbose',
        'Content-Type': 'application/json'
    }
    
    test_url = f"{site_url}/_api/web"
    
    try:
        response = requests.get(test_url, headers=headers)
        if response.status_code == 200:
            site_data = response.json()
            site_title = site_data.get('d', {}).get('Title', 'Unknown')
            
            print(f"   ✅ REST API access successful")
            print(f"   ✅ Site title: {site_title}")
            return True
        else:
            print(f"   ❌ REST API failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ REST API exception: {e}")
        return False

if __name__ == "__main__":
    success = test_sharepoint_final()
    
    print("=" * 60)
    if success:
        print("🎉 SHAREPOINT ACCESS CONFIRMED!")
        print("✅ Your app registration is working properly")
        print("🚀 Ready to proceed with CSV file uploads")
        print()
        print("Next step: Run the actual upload script")
    else:
        print("❌ SHAREPOINT ACCESS STILL BLOCKED")
        print("💡 The app is registered but may need additional permissions")
        print()
        print("Recommendations:")
        print("1. Wait 5-10 minutes for permissions to propagate")
        print("2. Check if the app needs admin consent in Azure")
        print("3. Verify the app has Sites.ReadWrite.All permission")
    print("=" * 60)
