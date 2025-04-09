import numpy as np
import matplotlib.pyplot as plt

def U(x, y):
    return np.sin(x) * np.cos(y) + 2 * (x * y)**2 / 1000

def grad_U(x, y):
    dU_dx = np.cos(x) * np.cos(y) + 4 * x * y**2 / 1000
    dU_dy = -np.sin(x) * np.sin(y) + 4 * y * x**2 / 1000
    return np.array([dU_dx, dU_dy])

# Gradiente descendente para função bidimensional
def gradiente_descendente_2d(U, grad_U, r0, alpha=0.1, epsilon=1e-5, max_iter=1000):
    r_vals = [r0]
    U_vals = [U(*r0)]
    r = np.array(r0, dtype=float)
    for _ in range(max_iter):
        grad = grad_U(*r)
        r_new = r - alpha * grad
        r_vals.append(r_new.copy())
        U_vals.append(U(*r_new))
        if np.linalg.norm(r_new - r) < epsilon:
            break
        r = r_new
    return np.array(r_vals), U_vals

# Parâmetros iniciais
x0, y0 = 2, 2
alpha = 0.1
traj, U_values = gradiente_descendente_2d(U, grad_U, (x0, y0), alpha=alpha)

# Grade para gráfico de contorno
x = np.linspace(-4, 4, 400)
y = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x, y)
Z = U(X, Y)

# Plotando os gráficos
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico a: Contorno com trajetória
contour = axs[0].pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
axs[0].plot(traj[:, 0], traj[:, 1], 'r.-', label='Trajetória')
axs[0].set_title('Trajetória sobre o contorno de U(x, y)')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].legend()
fig.colorbar(contour, ax=axs[0])

# Gráfico b: Valor de U a cada passo (epoch)
axs[1].plot(U_values, 'b-o')
axs[1].set_title('Valor de U(x, y) por iteração')
axs[1].set_xlabel('Iterações (epochs)')
axs[1].set_ylabel('U(x, y)')
axs[1].grid()

plt.tight_layout()
