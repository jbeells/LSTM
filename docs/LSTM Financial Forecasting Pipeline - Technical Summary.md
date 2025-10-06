
# LSTM Financial Market Forecasting Pipeline

## Project Overview

This project implements a complete end-to-end machine learning pipeline for forecasting financial market indicators using LSTM (Long Short-Term Memory) neural networks. The system is designed for production deployment with automated daily execution, comprehensive monitoring, and integration with business intelligence tools.

## Architecture & Components

### 1. Core ML Model (`mkt_lstm.py`)

**Purpose**: Primary LSTM model for financial time series forecasting

**Key Features**:
- Fetches real-time financial data from FRED API (Federal Reserve Economic Data)
- Tracks four key market indicators:
  - S&P 500 Index (`SP500`)
  - VIX Volatility Index (`VIXCLS`) 
  - Dow Jones Industrial Average (`DJIA`)
  - High Yield Bond Index (`BAMLCC4A0710YTRIV`)
- Uses 10-day sequence length for LSTM input
- MinMaxScaler for data normalization
- 80/20 train/test split with comprehensive metrics

**Model Architecture**:
```
Input Layer → LSTM(64 units) → Dense(output_dim) → Output
```

**Outputs**:
- `actuals.csv`: Historical market data
- `predicted.csv`: Model predictions on historical data
- `forecasted_data.csv`: 30-day forward forecasts
- `model_metrics.csv`: Performance metrics (MSE, MAE, R²)
- `model_history.csv`: Training loss progression
- Saved model artifacts (`lstm_model.keras`, `scaler.pkl`)

### 2. Daily Execution Pipeline (`daily_forecast.py`)

**Purpose**: Production-ready daily forecasting workflow

**Key Features**:
- **Market Calendar Integration**: Uses NYSE calendar to determine trading days
- **Robust Error Handling**: Comprehensive exception handling with detailed logging
- **Email Notifications**: Automated failure notifications via SMTP
- **Data Validation**: Ensures data quality and completeness
- **Performance Monitoring**: Calculates and tracks model metrics daily

**Workflow**:
1. Market status validation (weekends/holidays check)
2. FRED data refresh
3. Model and scaler loading
4. Recent year data processing (365 days)
5. Prediction generation
6. Performance metric calculation
7. 30-day forecast generation
8. Data validation and CSV output
9. Success/failure notification

### 3. Cloud Integration (`daily_blob_uploader.py`)

**Purpose**: Azure Blob Storage integration for data persistence

**Features**:
- Uploads daily outputs to Azure Blob Storage
- Timestamp-based folder organization (`YYYYMMDD_HHMMSS`)
- Supports containerized deployment
- Error handling and logging

### 4. CI/CD Automation (`daily_forecast.yaml`)

**Purpose**: GitHub Actions workflow for automated daily execution

**Schedule**: Runs weekdays at 9:30 AM EST (post-market open)

**Pipeline Steps**:
1. Environment setup (Python 3.11, dependencies)
2. Project structure verification
3. Daily forecast execution
4. Result validation and artifact upload
5. Azure Blob Storage deployment
6. Comprehensive logging and monitoring

**Security**: Uses GitHub Secrets for sensitive credentials (API keys, email, Azure)

### 5. Power BI Integration (`pbi_integration.md`)

**Purpose**: Business intelligence dashboard integration

**Implementation**:
- Dynamic Power Query integration with Azure Blob Storage
- Automatic latest folder discovery using date-based naming
- Real-time data refresh capabilities
- SAS token authentication for secure access

**Power Query Features**:
- `GetLatestFolder`: Discovers most recent data folder
- `GetDataFromLatestFolder`: Loads and combines CSV files
- Automatic data type conversion and validation
- Scheduled refresh integration

## Technical Specifications

### Data Pipeline
- **Data Source**: FRED API with automatic retry mechanisms
- **Data Frequency**: Daily (business days only)
- **Lookback Period**: 365 days for model input
- **Forecast Horizon**: 30 business days
- **Storage**: Azure Blob Storage with timestamp organization

### Model Performance
- **Sequence Length**: 10 days
- **Training Epochs**: 20 (configurable)
- **Validation Method**: Time series split (80/20)
- **Metrics**: MSE, MAE, R² for each market indicator
- **Model Persistence**: Keras format with pickle scaler

### Infrastructure
- **Compute**: GitHub Actions runners (Ubuntu)
- **Storage**: Azure Blob Storage
- **Monitoring**: Comprehensive logging with email alerts
- **BI Integration**: Power BI Service with automated refresh

## Deployment & Operations

### Environment Variables Required
```
FRED_API_KEY                    # Federal Reserve data access
EMAIL_USER, EMAIL_PASS         # SMTP notification credentials
EMAIL_HOST, EMAIL_PORT         # Email server configuration
NOTIFICATION_EMAIL             # Alert recipient
AZURE_STORAGE_CONNECTION_STRING # Azure storage access
AZURE_CONTAINER_NAME           # Blob storage container
```

### Directory Structure
```
project/
├── src/
│   ├── mkt_lstm.py           # Core ML model
│   ├── daily_forecast.py     # Daily execution script
│   └── daily_blob_uploader.py # Azure integration
├── models/                   # Saved model artifacts
├── data/
│   ├── output/              # Model training outputs
│   └── upload/              # Daily forecast outputs
├── logs/                    # Execution logs
└── .github/workflows/       # CI/CD configuration
```

### Monitoring & Alerting
- **Success Metrics**: CSV file generation, blob upload confirmation
- **Failure Detection**: Exception handling with detailed stack traces
- **Notifications**: Automated email alerts for failures
- **Logging**: Timestamped logs with rotation (30-90 day retention)

## Data Science Applications

This pipeline serves multiple use cases for the data science community:

1. **Research Platform**: Historical predictions vs. actuals for model validation
2. **Benchmarking**: Performance metrics tracking for model comparison
3. **Feature Engineering**: Multi-asset correlation analysis
4. **Risk Management**: Volatility forecasting and market regime detection
5. **Strategy Development**: Forward-looking market indicators for quantitative strategies

## Integration Points

### For Data Scientists
- **Model Access**: Pre-trained models available in Keras format
- **Data Access**: Clean, validated datasets in CSV format
- **Metrics Tracking**: Historical performance data for analysis
- **Extensibility**: Modular design for additional indicators/models

### For Business Users
- **Power BI Dashboard**: Real-time forecasts and performance metrics
- **Automated Updates**: Daily refresh with latest market data
- **Historical Analysis**: Trend analysis and model performance tracking

## Future Enhancement Opportunities

1. **Multi-Model Ensemble**: Integration of additional forecasting models
2. **Real-time Inference**: Intraday prediction capabilities
3. **Alternative Data**: Integration of sentiment, economic indicators
4. **MLOps Integration**: Model versioning, A/B testing, automated retraining
5. **Expanded Coverage**: Additional asset classes and markets

This pipeline represents a production-grade implementation of ML forecasting with enterprise-level reliability, monitoring, and integration capabilities.
