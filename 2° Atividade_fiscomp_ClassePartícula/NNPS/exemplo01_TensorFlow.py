import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Fixar a semente para reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

# ===========================
# GERANDO OS DADOS
# ===========================

# Dados de entrada
x = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
y = np.sin(x)

# Dividir em treino e teste
split = int(0.8 * len(x))
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

# ===========================
# CONSTRUINDO O MODELO
# ===========================

def criar_modelo(n_camadas=2, n_neuronios=64, ativacao='tanh'):
    modelo = tf.keras.Sequential()
    modelo.add(tf.keras.layers.Input(shape=(1,)))

    for _ in range(n_camadas):
        modelo.add(tf.keras.layers.Dense(n_neuronios, activation=ativacao))

    modelo.add(tf.keras.layers.Dense(1))  # camada de saída
    modelo.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss='mse')
    return modelo

# ===========================
# TREINAMENTO
# ===========================

modelo = criar_modelo(n_camadas=2, n_neuronios=64)

history = modelo.fit(x_train, y_train,
                     epochs=500,
                     batch_size=32,
                     validation_data=(x_test, y_test),
                     verbose=0)

# ===========================
# AVALIAÇÃO
# ===========================

# Predição
y_pred = modelo.predict(x)

# ===========================
# PLOTAGEM
# ===========================

plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Função seno (real)', linewidth=2)
plt.plot(x, y_pred, '--', label='Rede Neural (previsão)', linewidth=2)
plt.title('Interpolação da função seno usando TensorFlow')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
