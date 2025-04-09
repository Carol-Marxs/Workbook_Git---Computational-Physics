import numpy as np
import matplotlib.pylab as plt
import math

#funções

def U3(x):
    return x**2 * (x-1) * (x+1) + x / 4

def grad_U3(x):
    return 4 * x**3 - 2 * x + 1 / 4

#parametros do gradiente descendente

def gradiente_descendente(f, gra_f, x0, alpha = 0.01, epsilon = 0.001, max_iter = 1000):
    x_vals = [x0]
    x = x0
    for _ in range(max_iter):
        grad = gra_f(x)
        x_new = x - alpha * grad
        x_vals.append(x_new)
        if abs(x_new - x) < epsilon:
            break
        x = x_new
    return x_vals

#parametros da questão

x0 = 2 
alpha = 0.02
traj = gradiente_descendente(U3, grad_U3, x0, alpha)

#parametros de plot

x = np.linspace(-2, 2, 500)
plt.plot(x, U3(x), label= 'U(x)', color= 'blue')
plt.plot(traj, [U3(xi) for xi in traj], 'ro-', label='Trajetória')
plt.title('Gradiente Descendente - Exercício 3')
plt.xlabel('x')
plt.ylabel('U(x)')
plt.grid()
plt.legend()
plt.show()
