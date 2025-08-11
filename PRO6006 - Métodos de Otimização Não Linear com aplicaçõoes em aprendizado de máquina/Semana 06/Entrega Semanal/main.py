import torch
import numpy as np
from data_loader     import prepare_data
from models          import RNN, LSTM
from arima_benchmark import get_arima_predictions
from evaluation      import calculate_metrics, plot_results

def run_predictions(model, X_test, scaler):
    model.eval()
    predictions_scaled = []
    with torch.no_grad():
        for i in range(len(X_test)):
            seq = X_test[i]
            y_pred_scaled = model(seq)
            predictions_scaled.append(y_pred_scaled.item())
    
    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
    return predictions.flatten()


WINDOW_SIZE = 12  # Window size for sequential models (12 months)
FORECAST_HORIZONS = [1, 3, 12] # Horizons to predict: 1 month, 3 months, 12 months

HIDDEN_SIZES = [50, 25]

INPUT_SIZE    = 1     
OUTPUT_SIZE   = 1     
EPOCHS        = 200        
LEARNING_RATE = 0.001

for horizon in FORECAST_HORIZONS:
    X_train, y_train, X_test, y_test, scaler, df_full = prepare_data(
        window_size=WINDOW_SIZE, 
        forecast_horizon=horizon
    )

    y_test_actual = scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
    test_dates = df_full.index[-len(y_test_actual):]

    all_predictions = []
    model_names = ['ARIMA', 'Multi-Layer RNN', 'Multi-Layer LSTM']

    arima_preds, _ = get_arima_predictions(df_full, horizon)
    all_predictions.append(arima_preds[:len(y_test_actual)])
    
    # RNN Model
    rnn_model = RNN(
        input_size=INPUT_SIZE, 
        hidden_sizes=HIDDEN_SIZES,
        output_size=OUTPUT_SIZE
    )
    rnn_model = rnn_model.train_loop(X_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE)
    rnn_preds = run_predictions(rnn_model, X_test, scaler)
    all_predictions.append(rnn_preds)
    
    # LSTM Model
    lstm_model = LSTM(
        input_size=INPUT_SIZE, 
        hidden_sizes=HIDDEN_SIZES,
        output_size=OUTPUT_SIZE
    )
    lstm_model = lstm_model.train_loop(X_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE)
    lstm_preds = run_predictions(lstm_model, X_test, scaler)
    all_predictions.append(lstm_preds)

    print(f"\n--- Evaluation Metrics for Horizon: {horizon} ---")
    for i, name in enumerate(model_names):
        metrics = calculate_metrics(y_test_actual, all_predictions[i])
        print(f"  Model: {name}")
        print(f"    MSE: {metrics['MSE']:.4f}, MAE: {metrics['MAE']:.4f}, MAPE: {metrics['MAPE']:.2f}%")
    
    plot_results(test_dates, y_test_actual, all_predictions, model_names, horizon)