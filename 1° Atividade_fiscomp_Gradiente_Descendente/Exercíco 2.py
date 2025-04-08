import numpy as np
import matplotlib.pyplot as plt

#funções

def U2(x):
    return x**2 * (x - 1) * (x + 1)

def grad_U2(x):
    return 4 * x**3 - 2 * x

#parametros do gradiente descendente

def gradiente_descendente(f, grad_f, x0, alpha = 0.14, epsilon = 0.01, max_iter = 1000):
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

#parametros
x0 = 2
alpha = 0.14 
traj = gradiente_descendente(U2, grad_U2, x0, alpha)

# plot do Gráfico
x = np.linspace(-2, 2 , 50)
plt.plot(x, U2(x), label= 'x**2 * (x - 1) * (x + 1)', color='blue')
plt.plot(traj, [U2(xi) for xi in traj], 'yo-', label= 'Trajetória da partícula')
plt.title('Gradiente Descendente - Exercício 2')
plt.xlabel('x')
plt.ylabel(U2(x))
plt.grid()
plt.legend()
plt.show()
