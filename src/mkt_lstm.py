from fredapi import Fred
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
import os
import pickle
import warnings
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')

nyse = mcal.get_calendar('NYSE')

score_date = datetime.today().strftime('%Y-%m-%d')

# Base paths
PROJECT_ROOT = '/Users/jeells/Nextcloud/Projects/LSTM'
DATA_OUTPUT = os.path.join(PROJECT_ROOT, 'data/output')
MODEL_OUTPUT = os.path.join(PROJECT_ROOT, 'models')

os.makedirs(DATA_OUTPUT, exist_ok=True)
os.makedirs(MODEL_OUTPUT, exist_ok=True)

def update_fred_data(start_day='2020-01-01'):
    # Update FRED data, Read FRED API key from file
    try:
        fred = Fred(api_key=os.getenv('FRED_API_KEY'))
    except FileNotFoundError:
        raise FileNotFoundError("FRED API key file not found. Please ensure 'FRED_API_KEY' exists in the project root.")
    except Exception as e:
        raise Exception(f"Error initializing FRED API: {e}")

    try:
        sp_500 = fred.get_series('SP500', observation_start=start_day)
        vix = fred.get_series('VIXCLS', observation_start=start_day)
        djia = fred.get_series('DJIA', observation_start=start_day)
        bond = fred.get_series('BAMLCC4A0710YTRIV', observation_start=start_day)
    except Exception as e:
        raise Exception(f"Error fetching data from FRED API: {e}")

    df_sp500 = pd.DataFrame(sp_500, columns=['SP500'])
    df_sp500['Date'] = df_sp500.index

    df_vix = pd.DataFrame(vix, columns=['VIXCLS'])
    df_vix['Date'] = df_vix.index

    df_djia = pd.DataFrame(djia, columns=['DJIA'])
    df_djia['Date'] = df_djia.index

    df_bond = pd.DataFrame(bond, columns=['BAMLCC4A0710YTRIV'])
    df_bond['Date'] = df_bond.index
    df_bond = df_bond.rename(columns={'BAMLCC4A0710YTRIV': 'HY_BOND_IDX'})

    df_data = df_sp500.merge(df_vix, on='Date', how='left')
    df_data = df_data.merge(df_djia, on='Date', how='left')
    df_data = df_data.merge(df_bond, on='Date', how='left')
    df_data['Date'] = pd.to_datetime(df_data['Date'])
    df_data.set_index('Date', inplace=True)
    df_data = df_data.dropna()
    schedule = nyse.schedule(start_date=df_data.index.min(), end_date=df_data.index.max())
    df_data = df_data[df_data.index.isin(schedule.index)]
    return df_data

df_data = update_fred_data()

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_data)

# Prepare sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LEN = 10
X, y = create_sequences(scaled_data, SEQ_LEN)

# Split into train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQ_LEN, df_data.shape[1])),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(df_data.shape[1])
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model and capture history
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# Add Epoch column (1-indexed)
history_df = pd.DataFrame(history.history)
history_df.insert(0, 'Epoch', history_df.index + 1)

# Rename columns
history_df = history_df.rename(columns={
    'loss': 'Training_Loss',
    'val_loss': 'Test_Loss'
})

# Save training and validation loss to CSV
history_df.to_csv(os.path.join(DATA_OUTPUT, 'model_history.csv'), index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LSTM Training & Validation Loss')
plt.legend()
plt.savefig(os.path.join(DATA_OUTPUT, 'loss_curve.png'))
plt.close()

# Number of days you are forecasting
n_days = 30  # or whatever number you want

# Get the last date in your df_data
last_date = df_data.index[-1]

# Use NYSE calendar to get the next n_days business days
future_dates = nyse.valid_days(start_date=last_date + pd.Timedelta(days=1), end_date=last_date + pd.Timedelta(days=60))
future_dates = future_dates[:n_days]

# Predict
y_pred = model.predict(X_test)
y_pred_all = model.predict(np.concatenate((X_train, X_test), axis=0))
y_pred_all_rescaled = scaler.inverse_transform(y_pred_all)
y_train_rescaled = scaler.inverse_transform(y_train) # Rescale y_train for completeness

y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# Convert predictions to DataFrame with correct column names
predicted_df = pd.DataFrame(y_pred_all_rescaled, columns=df_data.columns)
print(predicted_df.head())
print(predicted_df.shape)

predicted_df['Date'] = df_data.index[SEQ_LEN:SEQ_LEN + len(predicted_df)]
predicted_df = predicted_df.set_index('Date')

metrics = {}
for i, col in enumerate(df_data.columns):
    mse = mean_squared_error(y_test_rescaled[:, i], y_pred_rescaled[:, i])
    mae = mean_absolute_error(y_test_rescaled[:, i], y_pred_rescaled[:, i])
    r2 = r2_score(y_test_rescaled[:, i], y_pred_rescaled[:, i])
    metrics[col] = {'MSE': mse, 'MAE': mae, 'R2': r2}

# Flatten the dictionary for columns as keys and a single row of values
mod_metrics = {'Date': score_date}
for asset, m in metrics.items():
    for metric, value in m.items():
        mod_metrics[f'{asset}_{metric}'] = value

# DataFrame with keys as header and single row of values
metrics_df = pd.DataFrame([mod_metrics])
metrics_df.to_csv(os.path.join(DATA_OUTPUT, 'model_metrics.csv'), index=False)

def forecast_n_days(model, scaler, df_data, n_days=5):
    seq = scaler.transform(df_data)[-SEQ_LEN:].copy()
    forecasts = []
    for _ in range(n_days):
        pred_scaled = model.predict(seq.reshape(1, SEQ_LEN, df_data.shape[1]))
        pred = scaler.inverse_transform(pred_scaled)[0]
        forecasts.append(pred)
        seq = np.vstack([seq[1:], scaler.transform(pred.reshape(1, -1))])
    return np.array(forecasts)

# Generate n_days forecasts for future dates
forecasts = forecast_n_days(model, scaler, df_data, n_days=n_days)
forecast_df = pd.DataFrame(forecasts, columns=df_data.columns, index=future_dates)
forecast_df = forecast_df.reset_index().rename(columns={'index': 'Date'})

# Filter df_data to current year-to-date (YTD)
rolling_window_days = 365  # or 252 for trading days

# End date is today; start date is 1 year ago
end_date = pd.Timestamp(datetime.today())
start_date = end_date - pd.Timedelta(days=rolling_window_days)

df_data_rolling = df_data[(df_data.index >= start_date) & (df_data.index <= end_date)].copy()
pred_df_rolling = predicted_df[(predicted_df.index >= start_date) & (predicted_df.index <= end_date)].copy()

# Add a column to distinguish actuals vs. predicted, forecast
df_data_rolling['Type'] = 'Actual'
pred_df_rolling['Type'] = 'Predicted'
forecast_df_with_flag = forecast_df.copy()
forecast_df_with_flag['Type'] = 'Forecast'
forecast_df_with_flag = forecast_df_with_flag.set_index('Date')

# Standardize all frames to have ['Date', columns..., 'Type']
df_data_rolling = df_data_rolling.reset_index()
pred_df_rolling = pred_df_rolling.reset_index()
forecast_df_with_flag = forecast_df_with_flag.reset_index()

# Only keep columns that exist in all three DataFrames and 'Type'
all_value_cols = [col for col in df_data.columns if col in pred_df_rolling.columns and col in forecast_df_with_flag.columns]

# Select those columns + Date + Type
df_data_rolling = df_data_rolling[['Date'] + all_value_cols + ['Type']]
pred_df_rolling = pred_df_rolling[['Date'] + all_value_cols + ['Type']]
forecast_df_with_flag = forecast_df_with_flag[['Date'] + all_value_cols + ['Type']]

# In your Step 1 fix, after reset_index() calls:
df_data_rolling['Date'] = pd.to_datetime(df_data_rolling['Date']).dt.tz_localize(None)
pred_df_rolling['Date'] = pd.to_datetime(pred_df_rolling['Date']).dt.tz_localize(None)
forecast_df_with_flag['Date'] = pd.to_datetime(forecast_df_with_flag['Date']).dt.tz_localize(None)

# Concatenate
combined_df = pd.concat([df_data_rolling, pred_df_rolling, forecast_df_with_flag], axis=0, ignore_index=True)

# Check for each type
for type_name in ['Actual', 'Predicted', 'Forecast']:
    subset = combined_df[combined_df['Type'] == type_name]
    if not subset.empty:
        print(f"{type_name}: {len(subset)} rows, dates from {subset['Date'].min()} to {subset['Date'].max()}")
    else:
        print(f"{type_name}: No data found!")

# Ensure Date is datetime
combined_df['Date'] = pd.to_datetime(combined_df['Date'])

# Create the plots
fig, axes = plt.subplots(len(df_data.columns), 1, figsize=(16, 5 * len(df_data.columns)), sharex=True)
if len(df_data.columns) == 1:
    axes = [axes]  # Ensure axes is iterable for a single column

for i, col in enumerate(df_data.columns):
    # Filter by type
    actuals = combined_df[combined_df['Type'] == 'Actual'].copy()
    predicted = combined_df[combined_df['Type'] == 'Predicted'].copy()
    forecasts = combined_df[combined_df['Type'] == 'Forecast'].copy()
    
    # Sort by date for proper line plotting
    actuals = actuals.sort_values('Date')
    predicted = predicted.sort_values('Date')
    forecasts = forecasts.sort_values('Date')
    
    # Plot each series
    if not actuals.empty:
        axes[i].plot(actuals['Date'], actuals[col], label='Actual', color='blue', linewidth=3)
    
    if not predicted.empty:
        axes[i].plot(predicted['Date'], predicted[col], label='Predicted', color='orange', linewidth=2)
    
    if not forecasts.empty:
        axes[i].plot(forecasts['Date'], forecasts[col], label='Forecast', color='red', linestyle='--', linewidth=3)
    
    axes[i].set_title(f'{col}: Actual, Predicted  & Forecast')
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel(col)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

fig.suptitle('Actuals, Predicted, and 30 Day-Ahead Forecast', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(DATA_OUTPUT, 'forecast_plot.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save the DataFrames to CSV
df_data.iloc[SEQ_LEN:].reset_index().to_csv(os.path.join(DATA_OUTPUT, 'historical_data.csv'), index=False)
predicted_df.reset_index().to_csv(os.path.join(DATA_OUTPUT, 'predicted_data.csv'), index=False)
forecast_df.to_csv(os.path.join(DATA_OUTPUT, 'forecasted_data.csv'), index=False)

# Save the model
model.save(os.path.join(MODEL_OUTPUT, 'lstm_model.keras'))

# Save the scaler
with open(os.path.join(MODEL_OUTPUT, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
