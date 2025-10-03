#!/usr/bin/env python3
"""
SharePoint CSV Uploader using Delegated Authentication - FIXED
Uses correct MSAL client types for different authentication flows
"""

import os
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from msal import PublicClientApplication, ConfidentialClientApplication
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sharepoint_delegated_uploader():
    """Upload CSV using delegated permissions (user context)"""
    
    print("🚀 SHAREPOINT CSV UPLOADER - DELEGATED AUTH (FIXED)")
    print("=" * 60)
    print(f"⏰ Upload time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    tenant_id = os.getenv('SHAREPOINT_TENANT_ID')
    client_secret = os.getenv('SHAREPOINT_CLIENT_SECRET')
    client_id = "26a6a10b-5718-440c-8d1b-699fc88f7057"
    site_url = os.getenv('SHAREPOINT_SITE_URL', 'https://jeanalytics.sharepoint.com/sites/LSTM')
    
    if not all([tenant_id, client_secret]):
        print("❌ Missing environment variables")
        return False
    
    print("📋 CONFIGURATION")
    print("-" * 20)
    print(f"Client ID: {client_id}")
    print(f"Tenant: {tenant_id}")
    print(f"Site: {site_url}")
    print()
    
    # Step 1: Get delegated token
    print("🔑 STEP 1: AUTHENTICATION")
    print("-" * 30)
    
    token = get_delegated_token_fixed(tenant_id, client_id, client_secret)
    if not token:
        print("❌ Authentication failed")
        return False
    
    print("✅ Authentication successful")
    print()
    
    # Step 2: Get site information
    print("🏢 STEP 2: SITE ACCESS")
    print("-" * 25)
    
    site_id = get_site_id(token, site_url)
    if not site_id:
        print("❌ Site access failed")
        return False
    
    print(f"✅ Site access successful")
    print(f"Site ID: {site_id}")
    print()
    
    # Step 3: Get drive information
    print("📁 STEP 3: DRIVE ACCESS")
    print("-" * 26)
    
    drive_id = get_drive_id(token, site_id)
    if not drive_id:
        print("❌ Drive access failed")
        return False
    
    print(f"✅ Drive access successful")
    print(f"Drive ID: {drive_id}")
    print()
    
    # Step 4: Generate forecast data
    print("📊 STEP 4: DATA GENERATION")
    print("-" * 29)
    
    df = generate_forecast_data()
    print(f"✅ Generated {len(df)} forecast records")
    print("Sample data:")
    print(df.head().to_string(index=False))
    print()
    
    # Step 5: Upload CSV file
    print("📤 STEP 5: FILE UPLOAD")
    print("-" * 24)
    
    success = upload_csv_file(token, drive_id, df)
    
    if success:
        print("🎉 CSV UPLOAD SUCCESSFUL!")
        print("✅ File uploaded to SharePoint")
        return True
    else:
        print("❌ CSV upload failed")
        return False

def get_delegated_token_fixed(tenant_id, client_id, client_secret):
    """Get token using correct MSAL approach for delegated permissions"""
    
    authority = f"https://login.microsoftonline.com/{tenant_id}"
    scopes = ["https://graph.microsoft.com/Sites.ReadWrite.All"]
    
    # Method 1: Try Authorization Code Flow (best for automation)
    print("   Attempting authorization code flow...")
    token = try_authorization_code_flow(tenant_id, client_id, client_secret, scopes)
    if token:
        return token
    
    # Method 2: Try Device Code Flow with PublicClientApplication
    print("   Attempting device code flow...")
    token = try_device_code_flow(tenant_id, client_id, scopes)
    if token:
        return token
    
    # Method 3: Try Resource Owner Password Credentials (if enabled)
    print("   Attempting ROPC flow...")
    token = try_ropc_flow(tenant_id, client_id, client_secret, scopes)
    if token:
        return token
    
    return None

def try_authorization_code_flow(tenant_id, client_id, client_secret, scopes):
    """Try authorization code flow - best for web apps"""
    
    try:
        app = ConfidentialClientApplication(
            client_id=client_id,
            client_credential=client_secret,
            authority=f"https://login.microsoftonline.com/{tenant_id}"
        )
        
        # Check for cached token first
        accounts = app.get_accounts()
        if accounts:
            print("      Found cached account, trying silent auth...")
            result = app.acquire_token_silent(scopes, account=accounts[0])
            if result and "access_token" in result:
                print("      ✅ Silent authentication successful")
                return result["access_token"]
        
        # For automation, we need a different approach
        # Let's try the client credentials flow with different scopes
        print("      Trying client credentials with user impersonation...")
        
        # This won't work for delegated permissions, but let's see the error
        result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
        if result and "access_token" in result:
            print("      ✅ Client credentials successful")
            return result["access_token"]
        else:
            print(f"      ❌ Client credentials failed: {result.get('error_description', 'Unknown')}")
            
    except Exception as e:
        print(f"      ❌ Authorization code exception: {e}")
    
    return None

def try_device_code_flow(tenant_id, client_id, scopes):
    """Try device code flow with PublicClientApplication"""
    
    try:
        # Use PublicClientApplication for device code flow
        app = PublicClientApplication(
            client_id=client_id,
            authority=f"https://login.microsoftonline.com/{tenant_id}"
        )
        
        # Check for cached token first
        accounts = app.get_accounts()
        if accounts:
            print("      Found cached account, trying silent auth...")
            result = app.acquire_token_silent(scopes, account=accounts[0])
            if result and "access_token" in result:
                print("      ✅ Silent authentication successful")
                return result["access_token"]
        
        # Start device code flow
        print("      Starting device code flow...")
        flow = app.initiate_device_flow(scopes=scopes)
        
        if "user_code" in flow:
            print(f"      📱 Please visit: {flow['verification_uri']}")
            print(f"      🔑 Enter code: {flow['user_code']}")
            print("      ⏳ Waiting for authentication (timeout: 15 minutes)...")
            
            # Wait for user to complete authentication
            result = app.acquire_token_by_device_flow(flow)
            
            if result and "access_token" in result:
                print("      ✅ Device flow authentication successful")
                return result["access_token"]
            else:
                print(f"      ❌ Device flow failed: {result.get('error_description', 'Unknown error')}")
        else:
            print("      ❌ Failed to initiate device flow")
            
    except Exception as e:
        print(f"      ❌ Device code exception: {e}")
    
    return None

def try_ropc_flow(tenant_id, client_id, client_secret, scopes):
    """Try Resource Owner Password Credentials flow"""
    
    # This requires username/password and is typically disabled
    # We'll skip this for security reasons
    print("      ROPC flow skipped (requires username/password)")
    return None

def get_site_id(token, site_url):
    """Get SharePoint site ID"""
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    # Extract site path from URL
    from urllib.parse import urlparse
    parsed_url = urlparse(site_url)
    site_path = parsed_url.path  # Should be /sites/LSTM
    
    # Try different Graph API formats
    api_urls = [
        f"https://graph.microsoft.com/v1.0/sites/{parsed_url.netloc}:{site_path}",
        f"https://graph.microsoft.com/v1.0/sites/{parsed_url.netloc}{site_path}",
        f"https://graph.microsoft.com/v1.0/sites?search=LSTM"
    ]
    
    for api_url in api_urls:
        try:
            print(f"   Trying: {api_url}")
            response = requests.get(api_url, headers=headers)
            print(f"   Response: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Handle search results
                if 'value' in data:
                    for site in data['value']:
                        if 'LSTM' in site.get('displayName', ''):
                            print(f"   Found site: {site.get('displayName')}")
                            return site.get('id')
                else:
                    # Direct site access
                    print(f"   Found site: {data.get('displayName', 'Unknown')}")
                    return data.get('id')
            else:
                print(f"   Error: {response.text[:200]}")
                
        except Exception as e:
            print(f"   Exception: {e}")
            continue
    
    return None

def get_drive_id(token, site_id):
    """Get default drive ID for the site"""
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive"
    
    try:
        print(f"   Getting drive: {drive_url}")
        response = requests.get(drive_url, headers=headers)
        print(f"   Drive response: {response.status_code}")
        
        if response.status_code == 200:
            drive_data = response.json()
            drive_name = drive_data.get('name', 'Unknown')
            print(f"   Drive name: {drive_name}")
            return drive_data.get('id')
        else:
            print(f"   Drive error: {response.text}")
    except Exception as e:
        print(f"   Drive exception: {e}")
    
    return None

def generate_forecast_data():
    """Generate sample LSTM forecast data"""
    
    import numpy as np
    
    # Generate 30 days of forecast data
    base_date = datetime.now().date()
    dates = [base_date + timedelta(days=i) for i in range(30)]
    
    # Generate realistic forecast values with trend and seasonality
    np.random.seed(42)  # For reproducible results
    
    trend = np.linspace(100, 120, 30)  # Upward trend
    seasonality = 10 * np.sin(np.linspace(0, 4*np.pi, 30))  # Weekly seasonality
    noise = np.random.normal(0, 5, 30)  # Random noise
    
    values = trend + seasonality + noise
    
    # Create confidence intervals
    lower_bounds = values - np.random.uniform(5, 15, 30)
    upper_bounds = values + np.random.uniform(5, 15, 30)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Forecast_Value': np.round(values, 2),
        'Lower_Bound': np.round(lower_bounds, 2),
        'Upper_Bound': np.round(upper_bounds, 2),
        'Model_Version': '1.0',
        'Generated_At': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    return df

def upload_csv_file(token, drive_id, df):
    """Upload CSV file to SharePoint"""
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'text/csv'
    }
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"lstm_forecast_{timestamp}.csv"
    
    # Convert DataFrame to CSV string
    csv_content = df.to_csv(index=False)
    csv_bytes = csv_content.encode('utf-8')
    
    # Upload file using Graph API
    upload_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{filename}:/content"
    
    try:
        print(f"   Uploading to: {upload_url}")
        response = requests.put(upload_url, headers=headers, data=csv_bytes)
        
        print(f"   Upload response: {response.status_code}")
        
        if response.status_code in [200, 201]:
            file_data = response.json()
            file_name = file_data.get('name', 'Unknown')
            file_size = file_data.get('size', 'Unknown')
            web_url = file_data.get('webUrl', 'No URL')
            
            print(f"   ✅ File uploaded: {file_name}")
            print(f"   ✅ File size: {file_size} bytes")
            print(f"   ✅ SharePoint URL: {web_url}")
            
            return True
        else:
            print(f"   ❌ Upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Upload exception: {e}")
        return False

if __name__ == "__main__":
    success = sharepoint_delegated_uploader()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 SHAREPOINT UPLOAD COMPLETED!")
        print("✅ CSV file successfully uploaded")
        print("✅ Ready for daily automation")
    else:
        print("❌ SHAREPOINT UPLOAD FAILED")
        print("💡 Check the error messages above")
    print("=" * 60)
