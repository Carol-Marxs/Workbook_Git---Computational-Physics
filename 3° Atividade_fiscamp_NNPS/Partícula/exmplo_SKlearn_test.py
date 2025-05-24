import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

def generate_data(nx, qtde):
    x = np.linspace(-1, 1, nx).reshape(-1, 1)
    y = []
    dy = []
    
    for _ in range(qtde // 2):
        p = 7
        coeffs = np.random.randn(p + 1)
        polinomio = np.polyval(coeffs, x)
        noise = 0.01 * np.random.randn(len(x)).reshape(-1, 1)
        y.append(polinomio / np.max(np.abs(polinomio)) + noise)

        der = np.polyval(np.polyder(coeffs), x)
        noise = 0.01 * np.random.randn(len(x)).reshape(-1, 1)
        dy.append(der / np.max(np.abs(der)) + noise)
    
    for _ in range(qtde // 2):
        p = 4
        coeffs = np.random.randn(p + 1)
        polinomio = np.polyval(coeffs, x)
        noise = 0.01 * np.random.randn(len(x)).reshape(-1, 1)
        y.append(polinomio / np.max(np.abs(polinomio)) + noise)

        der = np.polyval(np.polyder(coeffs), x)
        noise = 0.01 * np.random.randn(len(x)).reshape(-1, 1)
        dy.append(der / np.max(np.abs(der)) + noise)

    y = np.hstack(y).T
    dy = np.hstack(dy).T
    return y, dy

# Gerar dados
y, dy = generate_data(50, 10000)

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(y, dy, test_size=0.2, random_state=42)

# Normalização
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

# Modelo MLP
model = MLPRegressor(
    hidden_layer_sizes=(50, 50),
    activation='tanh',
    solver='adam',
    max_iter=1000,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    tol=1e-10,
    n_iter_no_change=100,
    verbose=True,
    random_state=42
)

model.fit(X_train_scaled, y_train_scaled)

# Gráfico
plt.figure(figsize=(12, 4))
new_x = np.linspace(0, 1, y.shape[1]).reshape(1, -1)

# Teste 1 - seno
plt.subplot(131)
new_y = np.sin(2 * np.pi * new_x)
new_dy = np.cos(2 * np.pi * new_x)
new_y_scaled = scaler_X.transform(new_y)
predicted_scaled = model.predict(new_y_scaled)
predicted = scaler_y.inverse_transform(predicted_scaled)

plt.plot(new_x[0], new_y[0], label='Input', color='black')
plt.plot(new_x[0], new_dy[0], label='True dy/dx', color='blue')
plt.plot(new_x[0], predicted[0], label='Predicted dy/dx', color='red', linestyle='dashed')
plt.ylim(-1, 1)
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.legend()
plt.grid(True)

# Teste 2 - cosseno
plt.subplot(132)
new_y = np.cos(2 * np.pi * new_x)
new_dy = -np.sin(2 * np.pi * new_x)
new_y_scaled = scaler_X.transform(new_y)
predicted_scaled = model.predict(new_y_scaled)
predicted = scaler_y.inverse_transform(predicted_scaled)

plt.plot(new_x[0], new_y[0], label='Input', color='black')
plt.plot(new_x[0], new_dy[0], label='True dy/dx', color='blue')
plt.plot(new_x[0], predicted[0], label='Predicted dy/dx', color='red', linestyle='dashed')
plt.ylim(-1, 1)
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.legend()
plt.grid(True)

# Teste 3 - função quadrática
plt.subplot(133)
new_y = new_x ** 2
new_dy = 2 * new_x
new_y_scaled = scaler_X.transform(new_y)
predicted_scaled = model.predict(new_y_scaled)
predicted = scaler_y.inverse_transform(predicted_scaled)

plt.plot(new_x[0], new_y[0], label='Input', color='black')
plt.plot(new_x[0], new_dy[0], label='True dy/dx', color='blue')
plt.plot(new_x[0], predicted[0], label='Predicted dy/dx', color='red', linestyle='dashed')
plt.ylim(-1, 1)
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
