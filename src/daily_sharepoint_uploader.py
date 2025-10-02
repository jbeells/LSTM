#!/usr/bin/env python3
"""
SharePoint CSV Uploader Script
==============================

Uploads CSV files from data/upload to SharePoint document library, overwriting existing files.
Uses SharePoint Online token-based authentication for better compatibility.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import requests
from requests.auth import HTTPBasicAuth
from requests.exceptions import RequestException
import json
import urllib.parse

# Setup logging to match your existing pattern
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_sharepoint_access_token():
    """Get SharePoint access token using legacy authentication"""
    username = os.getenv('SHAREPOINT_USERNAME')
    password = os.getenv('SHAREPOINT_PASSWORD')
    site_url = os.getenv('SHAREPOINT_SITE_URL')
    
    if not all([username, password, site_url]):
        raise ValueError("Missing required SharePoint credentials")
    
    # Parse tenant from site URL
    from urllib.parse import urlparse
    parsed_url = urlparse(site_url)
    tenant_name = parsed_url.hostname.split('.')[0]
    
    # SharePoint Online legacy authentication endpoint
    auth_url = f"https://login.microsoftonline.com/common/oauth2/token"
    
    # Try SAML authentication first
    try:
        return get_saml_token(site_url, username, password)
    except Exception as e:
        logger.warning(f"SAML auth failed: {e}")
        # Fallback to basic auth with better headers
        return get_basic_auth_session(username, password)

def get_saml_token(site_url, username, password):
    """Get SAML token for SharePoint authentication"""
    # SAML request template
    saml_template = """<?xml version="1.0" encoding="UTF-8"?>
    <s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:a="http://www.w3.org/2005/08/addressing" xmlns:u="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">
        <s:Header>
            <a:Action s:mustUnderstand="1">http://schemas.xmlsoap.org/ws/2005/02/trust/RST/Issue</a:Action>
            <a:ReplyTo><a:Address>http://www.w3.org/2005/08/addressing/anonymous</a:Address></a:ReplyTo>
            <a:To s:mustUnderstand="1">https://login.microsoftonline.com/extSTS.srf</a:To>
            <o:Security s:mustUnderstand="1" xmlns:o="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">
                <o:UsernameToken><o:Username>{username}</o:Username><o:Password>{password}</o:Password></o:UsernameToken>
            </o:Security>
        </s:Header>
        <s:Body>
            <t:RequestSecurityToken xmlns:t="http://schemas.xmlsoap.org/ws/2005/02/trust">
                <wsp:AppliesTo xmlns:wsp="http://schemas.xmlsoap.org/ws/2004/09/policy">
                    <a:EndpointReference><a:Address>{site_url}</a:Address></a:EndpointReference>
                </wsp:AppliesTo>
                <t:KeyType>http://schemas.xmlsoap.org/ws/2005/05/identity/NoProofKey</t:KeyType>
                <t:RequestType>http://schemas.xmlsoap.org/ws/2005/02/trust/Issue</t:RequestType>
                <t:TokenType>urn:oasis:names:tc:SAML:1.0:assertion</t:TokenType>
            </t:RequestSecurityToken>
        </s:Body>
    </s:Envelope>"""
    
    saml_request = saml_template.format(
        username=username,
        password=password,
        site_url=site_url
    )
    
    headers = {
        'Content-Type': 'application/soap+xml; charset=utf-8',
        'SOAPAction': 'http://schemas.xmlsoap.org/ws/2005/02/trust/RST/Issue'
    }
    
    response = requests.post(
        'https://login.microsoftonline.com/extSTS.srf',
        data=saml_request,
        headers=headers
    )
    
    if response.status_code == 200:
        # Extract token from response (simplified)
        # In a real implementation, you'd parse the XML response
        logger.info("SAML authentication successful")
        return create_session_with_saml_token(response.text, site_url)
    else:
        raise Exception(f"SAML authentication failed: {response.status_code}")

def get_basic_auth_session(username, password):
    """Create session with enhanced basic authentication"""
    session = requests.Session()
    session.auth = HTTPBasicAuth(username, password)
    
    # Add headers that might help with SharePoint Online
    session.headers.update({
        'Accept': 'application/json;odata=verbose',
        'Content-Type': 'application/json;odata=verbose',
        'User-Agent': 'Python SharePoint Client 1.0',
        'X-RequestForceAuthentication': 'true',
        'Authorization': f'Basic {requests.auth._basic_auth_str(username, password)}'
    })
    
    return session

def create_session_with_saml_token(saml_response, site_url):
    """Create session using SAML token (simplified implementation)"""
    session = requests.Session()
    session.headers.update({
        'Accept': 'application/json;odata=verbose',
        'Content-Type': 'application/json;odata=verbose',
        'User-Agent': 'Python SharePoint Client 1.0'
    })
    return session

def get_form_digest(session, site_url):
    """Get form digest value required for SharePoint POST operations"""
    try:
        digest_url = f"{site_url}/_api/contextinfo"
        
        # Try with different headers
        headers = {
            'Accept': 'application/json;odata=verbose',
            'Content-Type': 'application/json;odata=verbose',
            'X-RequestForceAuthentication': 'true'
        }
        
        response = session.post(digest_url, headers=headers)
        
        # Log more details for debugging
        logger.info(f"Form digest request status: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 401:
            logger.error("Authentication failed - check credentials")
            logger.error("Possible issues:")
            logger.error("1. Modern Authentication may be required")
            logger.error("2. Multi-Factor Authentication is enabled")
            logger.error("3. Legacy authentication is disabled")
            raise Exception("Authentication failed - credentials may be invalid or MFA required")
        
        response.raise_for_status()
        
        digest_data = response.json()
        return digest_data['d']['GetContextWebInformation']['FormDigestValue']
    
    except RequestException as e:
        logger.error(f"Failed to get form digest: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
            
            if e.response.status_code == 403:
                logger.error("SharePoint access forbidden. This usually means:")
                logger.error("1. User doesn't have proper permissions")
                logger.error("2. Site URL is incorrect")
                logger.error("3. Modern Authentication is required (App Registration needed)")
                logger.error("4. Legacy authentication is disabled on this tenant")
        raise
    except Exception as e:
        logger.error(f"Failed to get form digest: {e}")
        raise

def upload_file_to_sharepoint(session, site_url, folder_path, file_path, form_digest):
    """Upload a single file to SharePoint, overwriting if it exists"""
    try:
        file_name = file_path.name
        
        # Read file content
        with open(file_path, 'rb') as file:
            file_content = file.read()
        
        # URL encode the folder path properly
        encoded_folder_path = urllib.parse.quote(folder_path, safe='/')
        
        # SharePoint REST API endpoint for file upload
        upload_url = f"{site_url}/_api/web/GetFolderByServerRelativeUrl('{encoded_folder_path}')/Files/add(url='{file_name}',overwrite=true)"
        
        # Headers for file upload
        headers = {
            'Accept': 'application/json;odata=verbose',
            'X-RequestDigest': form_digest,
            'Content-Type': 'application/octet-stream'
        }
        
        logger.info(f"Uploading {file_name} to {upload_url}")
        
        response = session.post(upload_url, data=file_content, headers=headers)
        response.raise_for_status()
        
        logger.info(f"Successfully uploaded {file_name} to SharePoint")
        return True
        
    except RequestException as e:
        logger.error(f"Failed to upload {file_path.name}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Failed to upload {file_path.name}: {e}")
        raise

def upload_csv_files():
    """Upload all CSV files from data/upload to SharePoint"""
    
    try:
        # Get SharePoint configuration from environment
        site_url = os.getenv('SHAREPOINT_SITE_URL')
        folder_path = os.getenv('SHAREPOINT_FOLDER_PATH', '/sites/LSTM/Shared Documents')
        
        if not site_url:
            raise ValueError("Missing SHAREPOINT_SITE_URL in environment variables")
        
        logger.info(f"Connecting to SharePoint site: {site_url}")
        logger.info(f"Target folder: {folder_path}")
        
        # Initialize SharePoint session
        session = get_sharepoint_access_token()
        
        # Get form digest for authentication
        form_digest = get_form_digest(session, site_url)
        
        # Path to upload directory
        upload_dir = Path(os.path.dirname(__file__)) / 'data' / 'upload'
        
        if not upload_dir.exists():
            raise FileNotFoundError(f"Upload directory not found: {upload_dir}")
        
        # Find all CSV files
        csv_files = list(upload_dir.glob('*.csv'))
        
        if not csv_files:
            logger.warning("No CSV files found to upload")
            return True
        
        logger.info(f"Found {len(csv_files)} CSV files to upload")
        
        # Upload each file (this will overwrite existing files)
        for csv_file in csv_files:
            try:
                upload_file_to_sharepoint(session, site_url, folder_path, csv_file, form_digest)
                
            except Exception as e:
                logger.error(f"Failed to upload {csv_file.name}: {e}")
                raise
        
        logger.info(f"Successfully uploaded {len(csv_files)} files to SharePoint")
        return True
        
    except Exception as e:
        logger.error(f"SharePoint upload process failed: {e}")
        logger.error("Troubleshooting suggestions:")
        logger.error("1. Verify SharePoint site URL is correct")
        logger.error("2. Check user has edit permissions on the target folder")
        logger.error("3. Consider using App Registration instead of user credentials")
        logger.error("4. Verify folder path exists in SharePoint")
        return False

if __name__ == "__main__":
    success = upload_csv_files()
    sys.exit(0 if success else 1)
