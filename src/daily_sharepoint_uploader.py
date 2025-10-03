#!/usr/bin/env python3
"""
Comprehensive SharePoint diagnostic script
Get detailed error information and try alternative authentication methods
"""

import os
import requests
import json
from datetime import datetime
import base64

def diagnose_sharepoint_issues():
    """Comprehensive SharePoint diagnostics"""
    
    print("🔍 SHAREPOINT COMPREHENSIVE DIAGNOSTICS")
    print("=" * 60)
    print(f"⏰ Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    tenant_id = os.getenv('SHAREPOINT_TENANT_ID')
    client_secret = os.getenv('SHAREPOINT_CLIENT_SECRET')
    client_id = "26a6a10b-5718-440c-8d1b-699fc88f7057"
    site_url = os.getenv('SHAREPOINT_SITE_URL', 'https://jeanalytics.sharepoint.com/sites/LSTM')
    
    print(f"📋 Tenant: {tenant_id}")
    print(f"📋 Client: {client_id}")
    print(f"📋 Site: {site_url}")
    print()
    
    # Test 1: Token Analysis
    print("1️⃣ TOKEN ANALYSIS")
    print("-" * 30)
    
    graph_token = get_token_with_details(tenant_id, client_id, client_secret, 
                                       'https://graph.microsoft.com/.default')
    
    if graph_token:
        analyze_token(graph_token, "Microsoft Graph")
    
    # Test 2: Direct Site URL Variations
    print("\n2️⃣ SITE URL VARIATIONS TEST")
    print("-" * 30)
    
    if graph_token:
        test_site_url_variations(graph_token)
    
    # Test 3: Tenant Settings Check
    print("\n3️⃣ TENANT SETTINGS CHECK")
    print("-" * 30)
    
    if graph_token:
        check_tenant_settings(graph_token)
    
    # Test 4: Alternative Authentication
    print("\n4️⃣ ALTERNATIVE AUTHENTICATION")
    print("-" * 30)
    
    test_certificate_auth(tenant_id, client_id)
    
    # Test 5: Direct API Calls with Different Headers
    print("\n5️⃣ HEADER VARIATIONS TEST")
    print("-" * 30)
    
    if graph_token:
        test_header_variations(graph_token)
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print("Based on the diagnostics above, the likely causes are:")
    print("1. SharePoint tenant blocks app-only tokens")
    print("2. Site collection requires specific app registration")
    print("3. Additional consent required at tenant level")
    print("4. SharePoint Online Management Shell permission needed")
    
    # Provide next steps
    provide_next_steps()

def get_token_with_details(tenant_id, client_id, client_secret, scope):
    """Get token with detailed error information"""
    print(f"🔑 Getting token for scope: {scope}")
    
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
            token_info = response.json()
            access_token = token_info.get('access_token')
            expires_in = token_info.get('expires_in', 'Unknown')
            
            print(f"   ✅ Token obtained successfully")
            print(f"   ✅ Expires in: {expires_in} seconds")
            
            return access_token
        else:
            print(f"   ❌ Token failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Token exception: {e}")
        return None

def analyze_token(token, token_type):
    """Analyze JWT token contents"""
    print(f"🔍 Analyzing {token_type} token...")
    
    try:
        # JWT tokens have 3 parts: header.payload.signature
        parts = token.split('.')
        if len(parts) >= 2:
            # Decode payload (add padding if needed)
            payload = parts[1]
            payload += '=' * (4 - len(payload) % 4)  # Add padding
            decoded = base64.b64decode(payload)
            token_data = json.loads(decoded)
            
            print(f"   ✅ Token decoded successfully")
            print(f"   App ID: {token_data.get('appid', 'Not found')}")
            print(f"   Tenant: {token_data.get('tid', 'Not found')}")
            print(f"   Audience: {token_data.get('aud', 'Not found')}")
            
            # Check roles/scopes
            roles = token_data.get('roles', [])
            scopes = token_data.get('scp', '')
            
            if roles:
                print(f"   Roles: {', '.join(roles)}")
            if scopes:
                print(f"   Scopes: {scopes}")
                
        else:
            print("   ❌ Invalid JWT token format")
            
    except Exception as e:
        print(f"   ❌ Token analysis failed: {e}")

def test_site_url_variations(token):
    """Test different ways to access the site"""
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    # Different URL formats to try
    url_variations = [
        "https://graph.microsoft.com/v1.0/sites/jeanalytics.sharepoint.com:/sites/LSTM",
        "https://graph.microsoft.com/v1.0/sites/jeanalytics.sharepoint.com/sites/LSTM",
        "https://graph.microsoft.com/v1.0/sites?search=LSTM",
        "https://graph.microsoft.com/v1.0/sites/root/sites",
        "https://graph.microsoft.com/v1.0/sites/jeanalytics.sharepoint.com",
        "https://graph.microsoft.com/beta/sites/jeanalytics.sharepoint.com:/sites/LSTM"
    ]
    
    for i, url in enumerate(url_variations, 1):
        print(f"   Testing variation {i}: {url}")
        try:
            response = requests.get(url, headers=headers)
            print(f"      Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'value' in data:  # List of sites
                    print(f"      Found {len(data['value'])} sites")
                    for site in data['value'][:3]:  # Show first 3
                        print(f"         - {site.get('displayName', 'No name')}")
                elif 'displayName' in data:  # Single site
                    print(f"      Site: {data.get('displayName')}")
                    print(f"      ✅ SUCCESS! Site accessible")
                    return data
            elif response.status_code == 401:
                print(f"      ❌ Unauthorized")
            elif response.status_code == 403:
                print(f"      ❌ Forbidden")
            else:
                print(f"      Response: {response.text[:100]}...")
                
        except Exception as e:
            print(f"      ❌ Exception: {e}")
    
    return None

def check_tenant_settings(token):
    """Check tenant-level settings that might block app access"""
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    # Try to access organization info
    org_url = "https://graph.microsoft.com/v1.0/organization"
    
    try:
        response = requests.get(org_url, headers=headers)
        if response.status_code == 200:
            org_data = response.json()
            if 'value' in org_data and org_data['value']:
                org = org_data['value'][0]
                print(f"   ✅ Organization: {org.get('displayName', 'Unknown')}")
                print(f"   Tenant ID: {org.get('id', 'Unknown')}")
            else:
                print("   ❌ No organization data returned")
        else:
            print(f"   ❌ Organization check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Tenant check exception: {e}")

def test_certificate_auth(tenant_id, client_id):
    """Test if certificate authentication is required"""
    print("   Certificate authentication is an alternative method")
    print("   This would require uploading a certificate to Azure AD")
    print("   Skipping for now - use client secret method first")

def test_header_variations(token):
    """Test different header combinations"""
    
    base_url = "https://graph.microsoft.com/v1.0/sites/jeanalytics.sharepoint.com"
    
    header_variations = [
        {
            'name': 'Standard Headers',
            'headers': {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
        },
        {
            'name': 'With Accept Header',
            'headers': {
                'Authorization': f'Bearer {token}',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        },
        {
            'name': 'SharePoint Specific',
            'headers': {
                'Authorization': f'Bearer {token}',
                'Accept': 'application/json;odata=verbose',
                'Content-Type': 'application/json;odata=verbose'
            }
        }
    ]
    
    for variation in header_variations:
        print(f"   Testing: {variation['name']}")
        try:
            response = requests.get(base_url, headers=variation['headers'])
            print(f"      Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"      ✅ SUCCESS with {variation['name']}!")
                return True
            elif response.status_code == 401:
                error_text = response.text
                if "app only tokens" in error_text.lower():
                    print(f"      ❌ App-only tokens explicitly blocked")
                else:
                    print(f"      ❌ Unauthorized: {error_text[:100]}...")
                    
        except Exception as e:
            print(f"      ❌ Exception: {e}")
    
    return False

def provide_next_steps():
    """Provide actionable next steps based on diagnostics"""
    print("\n🚀 RECOMMENDED NEXT STEPS:")
    print("-" * 40)
    print()
    
    print("Option A - SharePoint Admin Settings:")
    print("1. Contact your SharePoint administrator")
    print("2. Ask them to enable 'App-Only Authentication' for your tenant")
    print("3. This is typically found in SharePoint Admin Center")
    print()
    
    print("Option B - Use Delegated Permissions:")
    print("1. Switch from Application to Delegated permissions")
    print("2. Use interactive login instead of client credentials")
    print("3. This requires user interaction but bypasses app-only restrictions")
    print()
    
    print("Option C - Service Account Approach:")
    print("1. Create a dedicated service account")
    print("2. Grant it SharePoint permissions")
    print("3. Use username/password authentication")
    print()
    
    print("Option D - PowerShell/CLI Alternative:")
    print("1. Use SharePoint Online Management Shell")
    print("2. Or Microsoft 365 CLI with different authentication")
    print("3. Upload files via PowerShell script")

if __name__ == "__main__":
    diagnose_sharepoint_issues()
