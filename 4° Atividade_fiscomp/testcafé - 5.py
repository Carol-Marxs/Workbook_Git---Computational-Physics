import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers

# --- Configurações do problema ---
r = 0.005      # Taxa de resfriamento [s⁻¹]
Tamb = 25      # Temperatura ambiente [°C]
T0 = 100       # Temperatura inicial [°C]
np.random.seed(42)

# --- Solução analítica ---
def analytical_solution(t):
    return Tamb + (T0 - Tamb) * np.exp(-r * t)

# --- Dados sintéticos com ruído no intervalo [0, 300s] ---
t_train = np.linspace(0, 300, 20).reshape(-1, 1)
T_train = analytical_solution(t_train) + np.random.normal(0, 0.5, size=t_train.shape)
T_train[0] = T0  # Garante T(0) = 100°C

# Normalização de entrada (tempo)
t_mean, t_std = t_train.mean(), t_train.std()
t_train_norm = (t_train - t_mean) / t_std

# --- Construção da rede neural ---
model = models.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(64, activation='tanh'),
    layers.Dense(64, activation='tanh'),
    layers.Dense(1)
])

# --- Função de perda customizada ---
@tf.function
def custom_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    # Penalização maior no início (t=0) e até t=300s
    t0_penalty = 100.0 * tf.square(y_true[0] - y_pred[0])  # T(0)
    return mse + t0_penalty

# --- Compilação e treino ---
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=custom_loss)
model.fit(t_train_norm, T_train, epochs=3000, verbose=0)

# --- Predição para t de 0 a 1000s ---
t_pred = np.linspace(0, 1000, 1000).reshape(-1, 1)
t_pred_norm = (t_pred - t_mean) / t_std
T_pred = model.predict(t_pred_norm, verbose=0)

# --- Correção no início para garantir T(0) = 100°C exatamente ---
T_pred += (T0 - T_pred[0])

# --- Solução analítica para comparação ---
T_exact = analytical_solution(t_pred)

# --- Gráfico final ---
plt.figure(figsize=(10, 6))
plt.plot(t_pred, T_exact, label='Solução Analítica', color='blue', linewidth=3)
plt.scatter(t_train, T_train, label='Dados de Treino', color='orange', s=80, zorder=5)
plt.plot(t_pred, T_pred, label='Rede Neural (Regressão)', color='green', linewidth=2.5)
plt.axvline(x=300, color='red', linestyle='--', label='Fim do Treino')
plt.xlabel('Tempo (s)', fontsize=12)
plt.ylabel('Temperatura (°C)', fontsize=12)
plt.title('Rede Neural ajustada com precisão até 300s', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)
plt.xlim(0, 1000)
plt.ylim(20, 110)
plt.tight_layout()
plt.show()
