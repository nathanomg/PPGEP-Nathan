import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations_max=1000, tol=1e-4):
        # Inicializa hiperparâmetros e variáveis internas
        self.learning_rate      = learning_rate
        self.n_iterations_max   = n_iterations_max
        self.tol                = tol
        self.n_iterations       = 0
        self.w                  = None
        self.b                  = None
        self.cost_history       = []

    def __sigmoid(self, z):
        # Função sigmoide
        return 1 / (1 + np.exp(-z))

    def _initialize(self, n_features):
        # Inicializa pesos, viés (bias) e variáveis de acompanhamento
        self.w = np.zeros(n_features)
        self.b = 0
        self.cost_history = []
        self.n_iterations = 0

    def predict_proba(self, X):
        # Prevê as probabilidades para as features de entrada X
        linear_model = np.dot(X, self.w) + self.b
        return self.__sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        # Prevê os rótulos binários para as features de entrada X
        probabilities = self.predict_proba(X)
        return [1 if i > threshold else 0 for i in probabilities]
    
    def _update_step(self, X, y, **kwargs):
        # Método abstrato para atualização de parâmetros (a ser implementado pelas subclasses)
        raise NotImplementedError("As subclasses devem implementar o método `_update_step`.")

    def fit(self, X, y, **kwargs):
        # Treina o modelo usando a estratégia de otimização escolhida
        n_samples, n_features = X.shape
        self._initialize(n_features)
        
        while self.n_iterations < self.n_iterations_max:
            self.n_iterations += 1
            
            # Atualiza os parâmetros do modelo (implementação da subclasse)
            self._update_step(X, y, **kwargs)

            # Calcula e armazena o custo (log-loss)
            epsilon = 1e-9
            y_pred_full = self.predict_proba(X)
            cost = (-1 / n_samples) * np.sum(y * np.log(y_pred_full + epsilon) + (1 - y) * np.log(1 - y_pred_full + epsilon))
            self.cost_history.append(cost)

            # Verifica a convergência
            if self.n_iterations > 1 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tol:
                print(f"Convergiu após {self.n_iterations} iterações.")
                break

# Batch Gradient Descent
class LR_BatchGradient(LogisticRegression):   
    def _update_step(self, X, y, **kwargs):
        # Realiza uma única atualização de gradiente descendente em lote
        n_samples = X.shape[0]
        y_predicted = self.predict_proba(X)
        
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
        db = (1 / n_samples) * np.sum(y_predicted - y)
        
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

# Stochastic (Mini-batch) Gradient Descent
class LR_StochasticGradient(LogisticRegression):
    def _update_step(self, X, y, **kwargs):
        # Realiza atualizações usando mini-lotes (batch_size=1 por padrão)
        batch_size = kwargs.get('batch_size', 1)
        n_samples = X.shape[0]
        
        # Embaralha os dados para atualizações estocásticas
        shuffled_indices = np.random.permutation(n_samples)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            y_predicted = self.predict_proba(X_batch)
            
            dw = (1 / batch_size) * np.dot(X_batch.T, (y_predicted - y_batch))
            db = (1 / batch_size) * np.sum(y_predicted - y_batch)
            
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

# Implementação do Otimizador Adam
class LR_Adam(LogisticRegression):
    def __init__(self, learning_rate=0.01, n_iterations_max=1000, tol=1e-5, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Inicializa hiperparâmetros específicos do Adam e variáveis internas
        super().__init__(learning_rate, n_iterations_max, tol)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m_w, self.v_w, self.m_b, self.v_b, self.t = 0, 0, 0, 0, 0

    def _initialize(self, n_features):
        # Inicializa pesos, viés (bias) e as estimativas de momento do Adam
        super()._initialize(n_features)
        self.m_w, self.v_w = np.zeros(n_features), np.zeros(n_features)
        self.m_b, self.v_b = 0, 0
        self.t = 0

    def _update_step(self, X, y, **kwargs):
        # Realiza atualizações de otimização Adam usando mini-lotes (batch_size=32 por padrão)
        batch_size = kwargs.get('batch_size', 32)
        n_samples = X.shape[0]
        
        # Embaralha os dados para atualizações com mini-lotes
        shuffled_indices = np.random.permutation(n_samples)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for i in range(0, n_samples, batch_size):
            self.t += 1
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            y_predicted = self.predict_proba(X_batch)
            dw = (1 / batch_size) * np.dot(X_batch.T, (y_predicted - y_batch))
            db = (1 / batch_size) * np.sum(y_predicted - y_batch)

            # Atualiza a estimativa do primeiro momento
            self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db

            # Atualiza a estimativa do segundo momento bruto
            self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw ** 2)
            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)

            # Calcula as estimativas de momento com correção de viés
            m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

            # Atualiza os parâmetros usando a regra do Adam
            self.w -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            self.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)