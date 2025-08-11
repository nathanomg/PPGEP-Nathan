from LogisticRegression      import LR_BatchGradient, LR_StochasticGradient, LR_Adam
from sklearn.datasets        import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import accuracy_score, confusion_matrix, precision_score
import matplotlib.pyplot as plt

"""
Este script compara três variantes de regressão logística (Gradiente em Batch, Gradiente Estocástico e Adam)
utilizando o conjunto de dados de câncer de mama da biblioteca scikit-learn.

Principais blocos do código:
- Carregamento e preparação dos dados: Carrega o dataset, separa em treino e teste, e aplica normalização.
- Inicialização dos modelos: Instancia três modelos de regressão logística com diferentes métodos de otimização.
- Treinamento e avaliação: Treina cada modelo, faz previsões, calcula métricas de desempenho (acurácia, precisão, matriz de confusão e custo final).
- Visualização da convergência: Plota a evolução da função de custo ao longo das iterações para cada modelo.
"""

# Carregamento do conjunto de dados de câncer de mama
cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target

# Separação dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Normalização dos dados
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Definição dos hiperparâmetros
learning_rate    = 0.01
n_iterations_max = 1000
tol              = 0.00001

# Inicialização dos modelos de regressão logística com diferentes métodos de otimização
models = [
    LR_BatchGradient(learning_rate, n_iterations_max, tol),
    LR_StochasticGradient(learning_rate, n_iterations_max, tol),
    LR_Adam(learning_rate, n_iterations_max, tol)
]

cost_histories = []
labels = ["Batch Gradient", "Stochastic Gradient", "Adam"]

# Treinamento, avaliação e coleta de métricas para cada modelo
for model, label in zip(models, labels):
    model.fit(X_train_scaled, y_train)
    y_predictions = model.predict(X_test_scaled)
    
    # Cross-validation
    accuracy   = accuracy_score(y_test, y_predictions)
    precision  = precision_score(y_test, y_predictions)
    final_cost = model.cost_history[-1] if model.cost_history else None
    cm         = confusion_matrix(y_test, y_predictions)

    print("-" * 30)
    print(f"{label} Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Final Cost: {final_cost:.6f}")
    print(f"{label} Confusion Matrix:\n{cm}")

    cost_histories.append(model.cost_history)

# Visualization of cost function convergence for each model
plt.figure()
for cost_history, label in zip(cost_histories, labels):
    plt.plot(range(len(cost_history)), cost_history, label=label)
plt.title("Cost Function Convergence")
plt.xlabel("Iteration")
plt.ylabel("Cost (Cross-Entropy Loss)")
plt.legend()
plt.grid(True)
plt.show()
