import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations_max=1000, tol=1e-4, verbose=False):
        self.learning_rate    = learning_rate
        self.n_iterations_max = n_iterations_max
        self.n_iterations     = 0
        self.tol              = tol
        self.w                = None
        self.b                = None
        self.cost_history     = []

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        linear_model = np.dot(X, self.w) + self.b
        return self.__sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return [1 if i > threshold else 0 for i in probabilities]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        grad_norm = np.inf
        self.w    = np.zeros(n_features)
        self.b    = 0
        self.cost_history = []
        
        while self.n_iterations <= self.n_iterations_max and \
              grad_norm > self.tol:

            self.n_iterations += 1
            y_predicted = self.predict_proba(X)
             
            epsilon = 1e-9
            cost = (-1 / n_samples) * np.sum(y * np.log(y_predicted + epsilon) + (1 - y) * np.log(1 - y_predicted + epsilon))
            self.cost_history.append(cost)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            grad_norm = np.linalg.norm(np.concatenate((dw, np.array([db]))))
            
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
