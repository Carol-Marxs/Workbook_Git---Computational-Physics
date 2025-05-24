import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

def generate_data(nx, qtde):
    x = np.linspace(-1, 1, nx).reshape(-1, 1)
    y = []
    dy = []

    for _ in range(qtde // 2):
        coeffs = np.random.randn(6)  # grau 5 para senos (ímpares)
        coeffs[::2] = 0  # zera termos pares
        polinomio = np.polyval(coeffs, x)
        noise = 0.1 * np.random.randn(len(x), 1)
        norm_poli = polinomio / np.max(np.abs(polinomio))
        y.append(norm_poli + noise)

        der = np.polyval(np.polyder(coeffs), x)
        noise = 0.1 * np.random.randn(len(x), 1)
        norm_der = der / np.max(np.abs(der))
        dy.append(norm_der + noise)

    for _ in range(qtde // 2):
        coeffs = np.random.randn(5)  # grau 4 para cossenos (pares)
        coeffs[1::2] = 0  # zera termos ímpares
        polinomio = np.polyval(coeffs, x)
        noise = 0.1 * np.random.randn(len(x), 1)
        norm_poli = polinomio / np.max(np.abs(polinomio))
        y.append(norm_poli + noise)

        der = np.polyval(np.polyder(coeffs), x)
        noise = 0.1 * np.random.randn(len(x), 1)
        norm_der = der / np.max(np.abs(der))
        dy.append(norm_der + noise)

    y = np.hstack(y).T
    dy = np.hstack(dy).T
    return y, dy

# gerar dados
y, dy = generate_data(50, 2000)

# escalonadores
scaler_y = StandardScaler()
scaler_dy = StandardScaler()

# normaliza dados
y_scaled = scaler_y.fit_transform(y)
dy_scaled = scaler_dy.fit_transform(dy)

# separa treino/teste
X_train, X_test, y_train, y_test = train_test_split(y_scaled, dy_scaled, test_size=0.2, random_state=42)

# define modelo
model = MLPRegressor(
    hidden_layer_sizes=(10,) * 5,
    activation='relu',
    solver='adam',
    max_iter=5000,
    tol=1e-10,
    n_iter_no_change=200,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    random_state=42,
    verbose=True
)

# treino
model.fit(X_train, y_train)

# teste
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# gráficos
y_label = np.linspace(0, 1, y.shape[1]).reshape(1, -1)
plt.figure(figsize=(12, 4))

# Teste 1: seno (ímpares)
n = 3
new_y = np.sin(n * np.pi * y_label)
new_dy = n * np.pi * np.cos(n * np.pi * y_label)
new_y_scaled = scaler_y.transform(new_y)
pred_scaled = model.predict(new_y_scaled)
pred = scaler_dy.inverse_transform(pred_scaled)

plt.subplot(131)
plt.plot(y_label[0], new_y[0], label='Input', color='black')
plt.plot(y_label[0], new_dy[0], label='True dy/dx', color='blue')
plt.plot(y_label[0], pred[0], label='Predicted dy/dx', color='red', linestyle='dashed')
plt.ylim(-2, 2)
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.grid(True)
plt.legend()

# Teste 2: cosseno (pares)
n = 4
new_y = np.cos(n * np.pi * y_label)
new_dy = -n * np.pi * np.sin(n * np.pi * y_label)
new_y_scaled = scaler_y.transform(new_y)
pred_scaled = model.predict(new_y_scaled)
pred = scaler_dy.inverse_transform(pred_scaled)

plt.subplot(132)
plt.plot(y_label[0], new_y[0], label='Input', color='black')
plt.plot(y_label[0], new_dy[0], label='True dy/dx', color='blue')
plt.plot(y_label[0], pred[0], label='Predicted dy/dx', color='red', linestyle='dashed')
plt.ylim(-2, 2)
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.grid(True)
plt.legend()

# Teste 3: parabólica
new_y = y_label ** 2
new_dy = 2 * y_label
new_y_scaled = scaler_y.transform(new_y)
pred_scaled = model.predict(new_y_scaled)
pred = scaler_dy.inverse_transform(pred_scaled)

plt.subplot(133)
plt.plot(y_label[0], new_y[0], label='Input', color='black')
plt.plot(y_label[0], new_dy[0], label='True dy/dx', color='blue')
plt.plot(y_label[0], pred[0], label='Predicted dy/dx', color='red', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
