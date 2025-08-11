from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def get_arima_predictions(df_full, forecast_horizon):
    train_data = df_full.loc[df_full.index < '2023-01-01']['ipca']
    test_data = df_full.loc[df_full.index >= '2023-01-01']['ipca']
    
    history = [x for x in train_data]
    predictions = []
    
    for t in range(len(test_data)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        
        output = model_fit.forecast(steps=forecast_horizon)
        
        yhat = output[-1]
        predictions.append(yhat)
        
        obs = test_data[t]
        history.append(obs)
        
    print("ARIMA benchmark forecasting complete.")
    return predictions, test_data.values