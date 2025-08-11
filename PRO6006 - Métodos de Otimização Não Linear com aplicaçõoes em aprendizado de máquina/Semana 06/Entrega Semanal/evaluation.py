import numpy as np
import matplotlib.pyplot as plt

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MSE": mse, "MAE": mae, "MAPE": mape}

def plot_results(test_dates, y_true, predictions, model_names, horizon):
    plt.figure(figsize=(15, 7))
    plt.plot(test_dates, y_true, label='Valores Reais (IPCA)', color='black', linewidth=2, marker='o')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, model_name in enumerate(model_names):
        plt.plot(test_dates, predictions[i], label=f'Previsão {model_name}', linestyle='--')

    plt.title(f'Previsão de Inflação (IPCA) - Horizonte: {horizon} Meses')
    plt.xlabel('Data')
    plt.ylabel('Variação Percentual (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()