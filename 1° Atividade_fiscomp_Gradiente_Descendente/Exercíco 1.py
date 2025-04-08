#              Exercício 1
import numpy as np
import matplotlib.pyplot as plt

# Função U(x) = x² - 1 e sua derivada
def U1(x):
    return x**2 - 1

def grad_U1(x):
    return 2 * x

# Algoritmo de gradiente descendente
def gradiente_descendente(f, grad_f, x0, alpha=0.1, epsilon=0.01, max_iter=1000):
    x_vals = [x0]
    x = x0
    for _ in range(max_iter):
        grad = grad_f(x)
        x_new = x - alpha * grad
        x_vals.append(x_new)
        if abs(x_new - x) < epsilon:
            break
        x = x_new
    return x_vals

# Parâmetros
x0 = 5
alpha = 0.1
traj = gradiente_descendente(U1, grad_U1, x0, alpha)

# Gráfico
x = np.linspace(-6, 6, 400)
plt.plot(x, U1(x), label='U1(x) = x² - 1', color='blue')
plt.plot(traj, [U1(xi) for xi in traj], 'ro-', label='Trajetória da partícula')
plt.title('Gradiente Descendente - Exercício 1')
plt.xlabel('x')
plt.ylabel('U1(x)')
plt.grid()
plt.legend()
plt.show()

    