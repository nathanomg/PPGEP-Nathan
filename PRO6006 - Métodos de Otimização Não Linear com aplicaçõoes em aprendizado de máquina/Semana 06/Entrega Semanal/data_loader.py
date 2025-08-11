import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import requests

def get_ipca_data(start_date='2013-01-01', end_date='2023-12-31'):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json&dataInicial={start_date.split('-')[2]}/{start_date.split('-')[1]}/{start_date.split('-')[0]}&dataFinal={end_date.split('-')[2]}/{end_date.split('-')[1]}/{end_date.split('-')[0]}"
    response = requests.get(url)
    data = response.json()
    
    df = pd.DataFrame(data)
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    df.set_index('data', inplace=True)
    df['valor'] = pd.to_numeric(df['valor'])
    df = df.rename(columns={'valor': 'ipca'})
    
    print("Successfully fetched IPCA data.")
    return df

def create_sequences(data, window_size, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size + forecast_horizon - 1])
    return np.array(X), np.array(y)

def prepare_data(window_size=12, forecast_horizon=1):
    df = get_ipca_data(start_date='2013-01-01', end_date='2023-12-31')

    ipca_series = df['ipca'].values
    
    train_data = df.loc[df.index < '2023-01-01']['ipca'].values
    test_data_full = df.loc[df.index >= '2023-01-01']['ipca'].values
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    
    all_data_scaled = scaler.transform(ipca_series.reshape(-1, 1)).flatten()

    X, y = create_sequences(all_data_scaled, window_size, forecast_horizon)
    
    train_cutoff = len(train_data) - window_size - forecast_horizon + 1
    
    X_train, y_train = X[:train_cutoff], y[:train_cutoff]
    X_test, y_test = X[train_cutoff:], y[train_cutoff:]
    
    X_train = torch.from_numpy(X_train).float().unsqueeze(-1)
    y_train = torch.from_numpy(y_train).float().unsqueeze(-1)
    X_test = torch.from_numpy(X_test).float().unsqueeze(-1)
    y_test = torch.from_numpy(y_test).float().unsqueeze(-1)

    print(f"Data prepared for horizon={forecast_horizon}, window={window_size}")
    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shape: X={X_test.shape}, y={y_test.shape}")

    return X_train, y_train, X_test, y_test, scaler, df