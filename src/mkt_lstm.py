import os
import pickle
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import tensorflow as tf
from dotenv import load_dotenv
from fredapi import Fred
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# --- Environment & Setup ---
load_dotenv()
warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')
nyse = mcal.get_calendar('NYSE')

SEQ_LEN = 10

# --- Functions ---
def update_fred_data(start_day='2020-01-01') -> pd.DataFrame:
    """Fetch SP500, VIX, DJIA, and HY Bond Index data from FRED API."""
    fred = Fred(api_key=os.getenv('FRED_API_KEY'))

    sp_500 = fred.get_series('SP500', observation_start=start_day)
    vix = fred.get_series('VIXCLS', observation_start=start_day)
    djia = fred.get_series('DJIA', observation_start=start_day)
    bond = fred.get_series('BAMLCC4A0710YTRIV', observation_start=start_day)

    df_sp500 = pd.DataFrame(sp_500, columns=['SP500'])
    df_sp500['Date'] = df_sp500.index
    df_vix = pd.DataFrame(vix, columns=['VIXCLS'])
    df_vix['Date'] = df_vix.index
    df_djia = pd.DataFrame(djia, columns=['DJIA'])
    df_djia['Date'] = df_djia.index
    df_bond = pd.DataFrame(bond, columns=['HY_BOND_IDX'])
    df_bond['Date'] = df_bond.index

    # Fill missing values by carrying forward the previous day's data
    df_sp500.fillna(method='ffill', inplace=True)
    df_vix.fillna(method='ffill', inplace=True)
    df_djia.fillna(method='ffill', inplace=True)
    df_bond.fillna(method='ffill', inplace=True)

    df_data = df_sp500.merge(df_vix, on='Date').merge(df_djia, on='Date').merge(df_bond, on='Date')
    df_data['Date'] = pd.to_datetime(df_data['Date'])
    df_data.fillna(method='ffill', inplace=True)

    schedule = nyse.schedule(start_date=df_data['Date'].min(), end_date=df_data['Date'].max())
    df_data = df_data[df_data['Date'].isin(schedule.index)]
    df_data = df_data.reset_index(drop=True)  # Ensures clean index
    return df_data

def create_sequences(data: np.ndarray, seq_length: int):
    """Prepare LSTM input-output sequences."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


def build_lstm(input_shape, output_dim: int):
    """Build and compile an LSTM model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def forecast_n_days(model, scaler, df_data: pd.DataFrame, n_days=5):
    """Forecast future values for n_days using a trained LSTM model."""
    seq = scaler.transform(df_data.drop('Date', axis=1))[-SEQ_LEN:].copy()
    forecasts = []
    for _ in range(n_days):
        pred_scaled = model.predict(seq.reshape(1, SEQ_LEN, seq.shape[1]), verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0]
        forecasts.append(pred)
        seq = np.vstack([seq[1:], scaler.transform(pred.reshape(1, -1))])
    return np.array(forecasts)


# --- Main Execution Block (only when run directly) ---
if __name__ == "__main__":
    # Base paths
    PROJECT_ROOT = os.getenv("PROJECT_ROOT", ".")
    DATA_OUTPUT = os.path.join(PROJECT_ROOT, 'data/output')
    MODEL_OUTPUT = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(DATA_OUTPUT, exist_ok=True)
    os.makedirs(MODEL_OUTPUT, exist_ok=True)

    score_date = datetime.today().strftime('%Y-%m-%d')

    # Fetch and prepare data
    df_data = update_fred_data()

    # Save actual data (include Date column explicitly)
    # Ensure Date is included as the first column
    cols_actual = ['Date'] + [c for c in df_data.columns if c != 'Date']
    df_data.to_csv(os.path.join(DATA_OUTPUT, 'actuals.csv'), columns=cols_actual, index=False, date_format='%Y-%m-%d')

    # Separate date column before scaling
    dates = df_data['Date']
    numeric_data = df_data.drop('Date', axis=1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    X, y = create_sequences(scaled_data, SEQ_LEN)
    split = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    model = build_lstm((SEQ_LEN, scaled_data.shape[1]), scaled_data.shape[1])
    history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, 'Epoch', history_df.index + 1)
    history_df.rename(columns={'loss': 'Training_Loss', 'val_loss': 'Test_Loss'}, inplace=True)
    history_df.to_csv(os.path.join(DATA_OUTPUT, 'model_history.csv'), index=False)

    # Save loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.title('LSTM Training & Validation Loss')
    plt.savefig(os.path.join(DATA_OUTPUT, 'loss_curve.png'))
    plt.close()

    # Evaluate predictions
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test)

    metrics = {}
    for i, col in enumerate(numeric_data.columns):
        metrics[col] = {
            'MSE': mean_squared_error(y_test_rescaled[:, i], y_pred_rescaled[:, i]),
            'MAE': mean_absolute_error(y_test_rescaled[:, i], y_pred_rescaled[:, i]),
            'R2': r2_score(y_test_rescaled[:, i], y_pred_rescaled[:, i])
        }

    metrics_row = {'Date': score_date}
    for asset, m in metrics.items():
        for metric, value in m.items():
            metrics_row[f'{asset}_{metric}'] = value
    pd.DataFrame([metrics_row]).to_csv(os.path.join(DATA_OUTPUT, 'model_metrics.csv'), index=False)

    # Generate predictions for all dates in df_data
    full_scaled = scaler.transform(numeric_data)
    X_full, _ = create_sequences(full_scaled, SEQ_LEN)
    preds_scaled = model.predict(X_full, verbose=0)
    preds = scaler.inverse_transform(preds_scaled)
    pred_dates = df_data['Date'].iloc[SEQ_LEN:].reset_index(drop=True)
    pred_df = pd.DataFrame(preds, columns=numeric_data.columns)
    pred_df.insert(0, 'Date', pred_dates)
    pred_df.to_csv(os.path.join(DATA_OUTPUT, 'predicts.csv'), index=False, date_format='%Y-%m-%d')

    # Forecast n_days ahead
    n_days = 30
    last_date = df_data['Date'].iloc[-1]
    future_dates = nyse.valid_days(start_date=last_date + pd.Timedelta(days=1), end_date=last_date + pd.Timedelta(days=60))[:n_days]

    forecasts = forecast_n_days(model, scaler, df_data, n_days)
    forecast_df = pd.DataFrame(forecasts, columns=numeric_data.columns, index=future_dates)
    forecast_df.reset_index().rename(columns={'index': 'Date'}).to_csv(os.path.join(DATA_OUTPUT, 'forecasts.csv'), index=False, date_format='%Y-%m-%d')

    # Save model and scaler
    model.save(os.path.join(MODEL_OUTPUT, 'lstm_model.keras'))
    with open(os.path.join(MODEL_OUTPUT, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
