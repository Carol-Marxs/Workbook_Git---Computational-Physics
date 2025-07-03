import numpy as np

# Parâmetros do problema
alpha = -1.0                              # Intensidade do potencial periódico
n_vals = np.array([-1, 0, 1])             # Índices dos vetores G
G_vals = 2 * np.pi * n_vals               # G_n = 2πn
N = len(G_vals)                           # Dimensão da matriz

# Valor fixo de k na 1ª zona de Brillouin
k = 0.5  # você pode escolher outro, como k = 0.0 ou k = np.pi/2

# Calcula os λ_{k - G} = |k - G|^2 (em unidades atômicas)
delta_k = k - G_vals
lambda_vals = delta_k**2

# Monta a matriz Hamiltoniana H(k)
H = np.zeros((N, N))
for i in range(N):
    H[i, i] = lambda_vals[i]  # termo cinético na diagonal
    if i > 0:
        H[i, i - 1] = alpha   # acoplamento com G_{i-1}
    if i < N - 1:
        H[i, i + 1] = alpha   # acoplamento com G_{i+1}

# Visualização da matriz
print("Matriz H(k) para k =", k)
print(H)
 # Calcula os autovalores (energias) e autovetores (coeficientes C)
eigenvalues, eigenvectors = np.linalg.eigh(H)

print("\nEnergias (autovalores) para k =", k)
print(np.sort(eigenvalues))



#uma ideia para aumentar: n_max = 2
# n_vals = np.arange(-n_max, n_max + 1) # Gera: [-2, -1, 0, 1, 2]