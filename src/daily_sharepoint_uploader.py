#!/usr/bin/env python3
"""
SharePoint CSV Uploader using Microsoft 365 CLI
This bypasses app-only authentication restrictions by using the M365 CLI
"""

import os
import pandas as pd
import subprocess
import json
import tempfile
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class M365SharePointUploader:
    """SharePoint uploader using Microsoft 365 CLI"""
    
    def __init__(self):
        self.site_url = os.getenv('SHAREPOINT_SITE_URL', 'https://jeanalytics.sharepoint.com/sites/LSTM')
        self.is_authenticated = False
        self.cli_available = False
    
    def check_cli_installation(self):
        """Check if Microsoft 365 CLI is installed"""
        logger.info("🔧 Checking Microsoft 365 CLI installation...")
        
        try:
            result = subprocess.run(['m365', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"   ✅ M365 CLI installed: {version}")
                self.cli_available = True
                return True
            else:
                logger.error("   ❌ M365 CLI not responding properly")
                return False
        except FileNotFoundError:
            logger.error("   ❌ Microsoft 365 CLI not found")
            logger.info("   💡 Install with: npm install -g @pnp/cli-microsoft365")
            return False
        except subprocess.TimeoutExpired:
            logger.error("   ❌ M365 CLI command timed out")
            return False
        except Exception as e:
            logger.error(f"   ❌ CLI check failed: {e}")
            return False
    
    def authenticate(self):
        """Authenticate with Microsoft 365"""
        if not self.cli_available:
            logger.error("M365 CLI not available")
            return False
        
        logger.info("🔑 Authenticating with Microsoft 365...")
        
        # Check if already logged in
        try:
            result = subprocess.run(['m365', 'status'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                status_data = json.loads(result.stdout)
                if status_data.get('connectedAs'):
                    logger.info(f"   ✅ Already authenticated as: {status_data.get('connectedAs')}")
                    self.is_authenticated = True
                    return True
        except Exception as e:
            logger.info(f"   Status check failed, proceeding with login: {e}")
        
        # Authenticate using device code flow
        logger.info("   Starting device code authentication...")
        try:
            result = subprocess.run(['m365', 'login', '--authType', 'deviceCode'], 
                                  capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                logger.info("   ✅ Authentication successful")
                self.is_authenticated = True
                return True
            else:
                logger.error(f"   ❌ Authentication failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("   ❌ Authentication timed out")
            return False
        except Exception as e:
            logger.error(f"   ❌ Authentication exception: {e}")
            return False
    
    def test_site_access(self):
        """Test if we can access the SharePoint site"""
        if not self.is_authenticated:
            logger.error("Not authenticated")
            return False
        
        logger.info("🏢 Testing SharePoint site access...")
        
        try:
            # Get site information
            result = subprocess.run(['m365', 'spo', 'site', 'get', 
                                   '--url', self.site_url, '--output', 'json'], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                site_data = json.loads(result.stdout)
                site_title = site_data.get('Title', 'Unknown')
                logger.info(f"   ✅ Site access successful: {site_title}")
                logger.info(f"   Site ID: {site_data.get('Id', 'Unknown')}")
                return True
            else:
                logger.error(f"   ❌ Site access failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"   ❌ Site access exception: {e}")
            return False
    
    def upload_csv_file(self, df, filename=None):
        """Upload CSV file to SharePoint document library"""
        if not self.is_authenticated:
            logger.error("Not authenticated")
            return False
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"lstm_forecast_{timestamp}.csv"
        
        logger.info(f"📤 Uploading file: {filename}")
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            df.to_csv(temp_file.name, index=False)
            temp_filepath = temp_file.name
        
        try:
            # Upload file to SharePoint
            # Default document library is usually "Shared Documents" or "Documents"
            library_name = "Shared Documents"  # Common default
            
            result = subprocess.run([
                'm365', 'spo', 'file', 'add',
                '--webUrl', self.site_url,
                '--folder', library_name,
                '--path', temp_filepath,
                '--name', filename,
                '--output', 'json'
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                file_data = json.loads(result.stdout)
                file_url = file_data.get('ServerRelativeUrl', 'Unknown')
                file_size = os.path.getsize(temp_filepath)
                
                logger.info(f"   ✅ File uploaded successfully")
                logger.info(f"   ✅ File path: {file_url}")
                logger.info(f"   ✅ File size: {file_size} bytes")
                logger.info(f"   ✅ SharePoint URL: {self.site_url}")
                
                return True
            else:
                # Try alternative document library names
                alternative_libraries = ["Documents", "DocumentLibrary", "Documenti"]
                
                for lib_name in alternative_libraries:
                    logger.info(f"   Trying alternative library: {lib_name}")
                    
                    alt_result = subprocess.run([
                        'm365', 'spo', 'file', 'add',
                        '--webUrl', self.site_url,
                        '--folder', lib_name,
                        '--path', temp_filepath,
                        '--name', filename,
                        '--output', 'json'
                    ], capture_output=True, text=True, timeout=120)
                    
                    if alt_result.returncode == 0:
                        file_data = json.loads(alt_result.stdout)
                        file_url = file_data.get('ServerRelativeUrl', 'Unknown')
                        
                        logger.info(f"   ✅ File uploaded to {lib_name}")
                        logger.info(f"   ✅ File path: {file_url}")
                        
                        return True
                
                logger.error(f"   ❌ Upload failed to all libraries: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"   ❌ Upload exception: {e}")
            return False
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_filepath)
            except:
                pass

def generate_forecast_data():
    """Generate sample LSTM forecast data"""
    logger.info("📊 Generating LSTM forecast data...")
    
    import numpy as np
    
    # Generate 30 days of forecast data
    base_date = datetime.now().date()
    dates = [base_date + timedelta(days=i) for i in range(30)]
    
    # Generate realistic forecast values
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
        'Lower_Bound_95': np.round(lower_bounds, 2),
        'Upper_Bound_95': np.round(upper_bounds, 2),
        'Model_Version': '1.2.0',
        'Training_Date': datetime.now().strftime('%Y-%m-%d'),
        'Generated_At': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Confidence_Score': np.round(np.random.uniform(0.85, 0.95, 30), 3)
    })
    
    logger.info(f"   ✅ Generated {len(df)} forecast records")
    
    return df

def main():
    """Main function to upload CSV using M365 CLI"""
    print("🚀 SHAREPOINT UPLOADER - MICROSOFT 365 CLI")
    print("=" * 55)
    print(f"⏰ Upload time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Step 1: Generate forecast data
        forecast_df = generate_forecast_data()
        print("Sample forecast data:")
        print(forecast_df.head().to_string(index=False))
        print()
        
        # Step 2: Initialize M365 uploader
        uploader = M365SharePointUploader()
        
        # Step 3: Check CLI installation
        if not uploader.check_cli_installation():
            print("\n❌ SETUP REQUIRED")
            print("Please install Microsoft 365 CLI:")
            print("1. Install Node.js if not already installed")
            print("2. Run: npm install -g @pnp/cli-microsoft365")
            print("3. Then run this script again")
            return False
        
        # Step 4: Authenticate
        print()
        if not uploader.authenticate():
            print("❌ Authentication failed")
            return False
        
        # Step 5: Test site access
        print()
        if not uploader.test_site_access():
            print("❌ Site access failed")
            return False
        
        # Step 6: Upload file
        print()
        if uploader.upload_csv_file(forecast_df):
            print("\n🎉 SHAREPOINT UPLOAD SUCCESSFUL!")
            print("✅ CSV file uploaded using M365 CLI")
            print("✅ This method bypasses app-only restrictions")
            return True
        else:
            print("\n❌ File upload failed")
            return False
            
    except Exception as e:
        logger.error(f"Main process failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    print("\n" + "=" * 55)
    if success:
        print("🎉 SUCCESS! Ready for automation")
        print("💡 This approach works even with app-only restrictions")
    else:
        print("❌ Upload failed - check error messages above")
    print("=" * 55)
