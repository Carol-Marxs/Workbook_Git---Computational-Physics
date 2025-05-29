import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Configurações do problema ---
Tamb = 25.0  # Temperatura ambiente [°C]
T0 = 100.0   # Temperatura inicial [°C]
np.random.seed(42)

# Solução analítica para comparação
def analytical_solution(t, r):
    return Tamb + (T0 - Tamb) * np.exp(-r * t)

# Dados sintéticos com ruído
t_train = np.linspace(0, 200, 10).reshape(-1, 1)
r_true = 0.005
T_train = analytical_solution(t_train, r_true) + np.random.normal(0, 0.5, size=t_train.shape)

# Rede PINN sem conhecer o valor de r
class PINN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = [
            layers.Dense(20, activation='tanh'),
            layers.Dense(20, activation='tanh'),
            layers.Dense(1)
        ]
        self.r = tf.Variable(0.01, trainable=True, dtype=tf.float32)  # Valor inicial para r

    def call(self, t):
        x = t
        for layer in self.hidden:
            x = layer(x)
        return x

# Conversão para tensores
t_train_tf = tf.convert_to_tensor(t_train, dtype=tf.float32)

# Inicialização do modelo
model = PINN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Função de perda combinada: erro nos dados e erro na física (PINN)
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        T_pred = model(t_train_tf)
        loss_data = tf.reduce_mean((T_pred - T_train)**2)

        # Física: dT/dt = r(Tamb - T)
        with tf.GradientTape() as tape2:
            tape2.watch(t_train_tf)
            T_pred_physics = model(t_train_tf)
        dTdt = tape2.gradient(T_pred_physics, t_train_tf)
        physics_residual = dTdt - model.r * (Tamb - T_pred_physics)
        loss_phys = tf.reduce_mean(physics_residual**2)

        loss = loss_data + loss_phys

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Treinamento
for epoch in range(5000):
    loss = train_step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.numpy():.6f}, r = {model.r.numpy():.6f}")

# Predição para avaliação
t_plot = np.linspace(0, 1000, 1000).reshape(-1, 1)
t_plot_tf = tf.convert_to_tensor(t_plot, dtype=tf.float32)
T_pred_plot = model(t_plot_tf).numpy()
T_analytical_plot = analytical_solution(t_plot, r_true)

# Plotagem
plt.figure(figsize=(10, 6))
plt.plot(t_plot, T_analytical_plot, label='Solução Analítica (r verdadeiro)', color='blue')
plt.plot(t_plot, T_pred_plot, label=f'PINN (r estimado = {model.r.numpy():.5f})', color='green')
plt.scatter(t_train, T_train, color='orange', label='Dados de treino')
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title('PINN estimando r a partir dos dados')
plt.legend()
plt.grid(True)
plt.show()
