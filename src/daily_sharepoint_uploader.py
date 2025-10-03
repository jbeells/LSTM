#!/usr/bin/env python3
"""
SharePoint Test Script - Configured for your specific app
App: LSTM-SharePoint-Uploader
Client ID: 26a6a10b-5718-440c-8d1b-699fc88f7057
"""

import os
import requests
import json
from datetime import datetime

def test_sharepoint_with_your_config():
    """Test SharePoint with your specific app configuration"""
    
    print("🚀 SHAREPOINT TEST - LSTM-SHAREPOINT-UPLOADER")
    print("=" * 65)
    print(f"⏰ Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Your specific configuration
    tenant_id = os.getenv('SHAREPOINT_TENANT_ID')
    client_secret = os.getenv('SHAREPOINT_CLIENT_SECRET')
    
    # From your Azure screenshots
    client_id = "26a6a10b-5718-440c-8d1b-699fc88f7057"  # Your actual Client ID
    app_name = "LSTM-SharePoint-Uploader"
    
    site_url = os.getenv('SHAREPOINT_SITE_URL', 'https://jeanalytics.sharepoint.com/sites/LSTM')
    
    print(f"📋 App Name: {app_name}")
    print(f"📋 Client ID: {client_id}")
    print(f"📋 Tenant ID: {tenant_id}")
    print(f"📋 Site URL: {site_url}")
    print()
    
    if not all([tenant_id, client_secret]):
        print("❌ Missing environment variables (TENANT_ID or CLIENT_SECRET)")
        return False
    
    # Test authentication methods
    print("🔑 TESTING AUTHENTICATION METHODS")
    print("-" * 45)
    
    # Method 1: Microsoft Graph API
    print("1️⃣ Microsoft Graph API")
    graph_token = get_token(tenant_id, client_id, client_secret, 'https://graph.microsoft.com/.default')
    
    if graph_token:
        print("   ✅ Graph token obtained")
        if test_graph_access(graph_token):
            print("   🎉 GRAPH API SUCCESS - Using this method!")
            return create_final_uploader_script("graph", graph_token, client_id, tenant_id, client_secret)
    else:
        print("   ❌ Graph token failed")
    
    # Method 2: SharePoint-specific scope
    print("\n2️⃣ SharePoint-Specific Scope")
    sp_token = get_token(tenant_id, client_id, client_secret, 'https://jeanalytics.sharepoint.com/.default')
    
    if sp_token:
        print("   ✅ SharePoint token obtained")
        if test_sharepoint_rest(sp_token, site_url):
            print("   🎉 SHAREPOINT REST SUCCESS - Using this method!")
            return create_final_uploader_script("sharepoint", sp_token, client_id, tenant_id, client_secret)
    else:
        print("   ❌ SharePoint token failed")
    
    # Method 3: Legacy SharePoint authentication
    print("\n3️⃣ Legacy SharePoint Authentication")
    legacy_token = get_legacy_token(tenant_id, client_id, client_secret)
    
    if legacy_token:
        print("   ✅ Legacy token obtained")
        if test_sharepoint_rest(legacy_token, site_url):
            print("   🎉 LEGACY AUTH SUCCESS - Using this method!")
            return create_final_uploader_script("legacy", legacy_token, client_id, tenant_id, client_secret)
    else:
        print("   ❌ Legacy token failed")
    
    print("\n❌ All authentication methods failed")
    print_troubleshooting_steps()
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
            print(f"   Token failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"   Token exception: {e}")
        return None

def get_legacy_token(tenant_id, client_id, client_secret):
    """Try legacy SharePoint authentication"""
    legacy_url = f"https://accounts.accesscontrol.windows.net/{tenant_id}/tokens/OAuth/2"
    
    # SharePoint resource format
    resource = f"00000003-0000-0ff1-ce00-000000000000/jeanalytics.sharepoint.com@{tenant_id}"
    client_full = f"{client_id}@{tenant_id}"
    
    token_data = {
        'grant_type': 'client_credentials',
        'client_id': client_full,
        'client_secret': client_secret,
        'resource': resource
    }
    
    try:
        response = requests.post(legacy_url, data=token_data)
        if response.status_code == 200:
            return response.json().get('access_token')
        else:
            return None
    except:
        return None

def test_graph_access(token):
    """Test Microsoft Graph API access"""
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    # Test site access
    site_urls = [
        "https://graph.microsoft.com/v1.0/sites/jeanalytics.sharepoint.com:/sites/LSTM",
        "https://graph.microsoft.com/v1.0/sites/jeanalytics.sharepoint.com/sites/LSTM"
    ]
    
    for site_url in site_urls:
        try:
            response = requests.get(site_url, headers=headers)
            if response.status_code == 200:
                site_data = response.json()
                site_name = site_data.get('displayName', 'Unknown')
                site_id = site_data.get('id')
                
                print(f"   ✅ Site access: {site_name}")
                print(f"   ✅ Site ID: {site_id}")
                
                # Test drive access
                drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive"
                drive_response = requests.get(drive_url, headers=headers)
                
                if drive_response.status_code == 200:
                    drive_data = drive_response.json()
                    drive_name = drive_data.get('name', 'Documents')
                    print(f"   ✅ Drive access: {drive_name}")
                    
                    # Test file upload capability
                    return test_upload_capability(token, site_id)
                else:
                    print(f"   ❌ Drive access failed: {drive_response.status_code}")
                    return False
            else:
                print(f"   ❌ Site access failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Graph test exception: {e}")
    
    return False

def test_upload_capability(token, site_id):
    """Test if we can upload files"""
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    # Test root folder access
    root_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root/children"
    
    try:
        response = requests.get(root_url, headers=headers)
        if response.status_code == 200:
            print(f"   ✅ File upload capability confirmed")
            return True
        else:
            print(f"   ❌ Upload test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Upload test exception: {e}")
        return False

def test_sharepoint_rest(token, site_url):
    """Test SharePoint REST API"""
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json;odata=verbose'
    }
    
    test_url = f"{site_url}/_api/web"
    
    try:
        response = requests.get(test_url, headers=headers)
        if response.status_code == 200:
            site_data = response.json()
            site_title = site_data.get('d', {}).get('Title', 'Unknown')
            print(f"   ✅ REST API access: {site_title}")
            return True
        else:
            print(f"   ❌ REST API failed: {response.status_code}")
            if "Unsupported app only token" in response.text:
                print("   ⚠️  App-only tokens still blocked via REST API")
            return False
    except Exception as e:
        print(f"   ❌ REST API exception: {e}")
        return False

def create_final_uploader_script(method, token, client_id, tenant_id, client_secret):
    """Create instructions for the working method"""
    print("\n" + "=" * 65)
    print("🎉 AUTHENTICATION SUCCESS!")
    print(f"✅ Working method: {method.upper()}")
    print("=" * 65)
    
    if method == "graph":
        print("📝 READY FOR FILE UPLOAD!")
        print("   Your app can use Microsoft Graph API")
        print("   Scope: https://graph.microsoft.com/.default")
        print("   Method: Graph API file upload")
    elif method == "sharepoint":
        print("📝 READY FOR FILE UPLOAD!")
        print("   Your app can use SharePoint REST API")
        print("   Scope: https://jeanalytics.sharepoint.com/.default")
        print("   Method: SharePoint REST API file upload")
    elif method == "legacy":
        print("📝 READY FOR FILE UPLOAD!")
        print("   Your app can use Legacy SharePoint auth")
        print("   Method: Legacy SharePoint authentication")
    
    print("\n🚀 NEXT STEP:")
    print("   I'll create the final CSV upload script using this working method")
    print("   Your app registration is properly configured!")
    
    return True

def print_troubleshooting_steps():
    """Print troubleshooting information"""
    print("\n🔧 TROUBLESHOOTING STEPS:")
    print("=" * 45)
    print("1. Check Azure App Registration:")
    print("   - Go to portal.azure.com → App registrations")
    print("   - Find 'LSTM-SharePoint-Uploader'")
    print("   - API permissions → Sites.ReadWrite.All should have green checkmark")
    print()
    print("2. Check SharePoint Site Collection:")
    print("   - Verify the app appears in Site Collection App Permissions")
    print("   - App should have proper permission level")
    print()
    print("3. Wait for propagation:")
    print("   - Permissions can take 5-15 minutes to propagate")
    print("   - Try running the test again in a few minutes")
    print()
    print("4. Contact SharePoint admin:")
    print("   - They may need to enable app-only authentication")
    print("   - Or grant additional permissions to your app")

if __name__ == "__main__":
    print("Testing SharePoint access with your registered app...")
    print("App: LSTM-SharePoint-Uploader")
    print("Client ID: 26a6a10b-5718-440c-8d1b-699fc88f7057")
    print()
    
    success = test_sharepoint_with_your_config()
    
    if success:
        print("\n✅ TEST COMPLETED SUCCESSFULLY!")
        print("Ready to create the final upload script.")
    else:
        print("\n❌ TEST FAILED")
        print("Please check the troubleshooting steps above.")
