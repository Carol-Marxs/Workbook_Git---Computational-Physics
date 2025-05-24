import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# --- 1. Gerar Dados de Treinamento ---
# Fixar sementes aleatórias para reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

# Gerar 100 ângulos aleatórios entre 0 e 4π
num_samples = 100
angles_train = np.random.uniform(0, 4 * np.pi, num_samples).reshape(-1, 1)

# Calcular o seno desses ângulos
sin_values_train = np.sin(angles_train)

# Adicionar ruído gaussiano para simular erro de medição
noise = np.random.normal(0, 0.1, sin_values_train.shape)
sin_values_train += noise

# --- 2. Definir o Modelo de Rede Neural ---
# Construímos uma Rede Neural Totalmente Conectada com três camadas ocultas de 10 neurônios cada
# A função de ativação é 'tanh', comum para suavidade em tarefas de regressão
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),                 # Uma entrada (ângulo)
    tf.keras.layers.Dense(10, activation='tanh'),      # Primeira camada oculta
    tf.keras.layers.Dense(10, activation='tanh'),      # Segunda camada oculta
    tf.keras.layers.Dense(10, activation='tanh'),      # Terceira camada oculta
    tf.keras.layers.Dense(1)                           # Camada de saída (valor escalar)
])

# Compilar o modelo com:
# - Otimizador Adam (taxa de aprendizado adaptativa)
# - Função de perda MSE (erro quadrático médio)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mse')

# --- 3. Treinar o Modelo ---
# Treinar o modelo com os dados de treinamento
# - epochs: número de épocas (passagens completas pelos dados)
# - batch_size: quantidade de amostras por atualização
# - verbose=0 esconde o progresso do treinamento
history = model.fit(angles_train, sin_values_train,
                    epochs=5000,
                    batch_size=16,
                    verbose=0)

# --- 4. Gerar Dados de Teste ---
# Criar 50 ângulos igualmente espaçados entre 0 e 6π para avaliação
angles_test = np.linspace(0, 6 * np.pi, 50).reshape(-1, 1)
sin_values_true = np.sin(angles_test)

# --- 5. Fazer Previsões ---
# Usar o modelo treinado para prever os valores do seno para os ângulos de teste
sin_values_predicted = model.predict(angles_test)

# --- 6. Visualizar os Resultados ---
plt.figure(figsize=(10, 6))

# Mostrar os dados de treinamento (com ruído)
plt.scatter(angles_train, sin_values_train, label='Dados de Treinamento (ruído)', alpha=0.5)

# Mostrar a função seno verdadeira
plt.plot(angles_test, sin_values_true, label='sen(x) verdadeiro', color='blue')

# Mostrar as previsões feitas pela rede neural
plt.plot(angles_test, sin_values_predicted, label='sen(x) previsto', color='red')

plt.xlabel('Ângulo (radianos)')
plt.ylabel('sen(x)')
plt.title('Interpolação da Função Seno com TensorFlow')
plt.legend()
plt.grid(True)
plt.show()

# --- 7. Relatar o Erro Quadrático Médio (MSE) ---
# O MSE dá uma medida quantitativa de quão próximas estão as previsões dos valores verdadeiros
mse = tf.reduce_mean(tf.square(sin_values_true - sin_values_predicted))
print(f"Erro Quadrático Médio nos Dados de Teste: {mse.numpy():.6f}")