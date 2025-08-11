from LogisticRegression      import LogisticRegression
from sklearn.datasets        import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import accuracy_score
import matplotlib.pyplot as plt

cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = LogisticRegression(learning_rate=0.01, n_iterations_max=1000, tol=0.001)
model.fit(X_train_scaled, y_train)

y_predictions = model.predict(X_test_scaled)
accuracy      = accuracy_score(y_test, y_predictions)


print("-" * 30)
print("Model parameters:")
print("-" * 30)
print("Weights (w):", model.w)
print("Bias (b):", model.b)

print("\n" + "-" * 30)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 30)

plt.figure()
plt.plot(range(model.n_iterations), model.cost_history)
plt.title("Cost Function Convergence")
plt.xlabel("Iteration")
plt.ylabel("Cost (Cross-Entropy Loss)")
plt.grid(True)
plt.show()
