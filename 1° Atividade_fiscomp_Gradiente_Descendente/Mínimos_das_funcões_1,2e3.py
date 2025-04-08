import numpy as np #importar funções ou constantes específicas do módulo 
import matplotlib.pyplot as plt #cria gráficos
import math #biblioteca matemática

# EXERCÍCIO_1_2_e_3

def U1(x): #EXERCÍCIO_1
    return x**2 - 1

def grad_U1(x): 
    return 2 * x

def U2(x):
    return x**2 * (x - 1) * (x + 1)

def grad_U2(x): #EXERCÍCIO_2
    return 4 * x**3 - 2 * x

def U3(x): #EXERCÍCIO_3
    return x**2 * (x - 1) * (x + 1) + x / 4 

def grad_U3(x): 
    return 4 * x**3 - 2 * x + 1 / 4 

# DADOS DA ATIVIDADE:

def gradiente_descendente(f, grad_f, x0, alpha=0.01, epsilon=0.01, max_iter=1000): 
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

x = np.linspace(-10,10,400)

x0 = 5
alpha = 0.01
traj_U1 = gradiente_descendente(U1, grad_U1, x0, alpha)
traj_U2 = gradiente_descendente(U2, grad_U2, x0, alpha)
traj_U3 = gradiente_descendente(U3, grad_U3, x0, alpha)

fig, axs = plt.subplots(3, 1, figsize=(8, 12))

#Para o EXERCÍCIO 1
axs[0].plot(x, U1(x), label='U(x) = x² - 1')
axs[0].plot(traj_U1, [U1(xi) for xi in traj_U1], 'ro-', label='Trajetória')
axs[0].set_title('Exercício 1: Gradiente Descendente em U(x) = x² - 1')
axs[0].legend()
axs[0].grid()

#Para o Exercício 2 
axs[0].plot(x, U2(x), label = 'x² (x - 1) * (x + 1)')
axs[0].plot(traj_U2, [U2(xi) for xi in traj_U2], 'go-', label='Trajetória')
axs[0].set_title('Exercício 2: Mútiplos Mínimos')
axs[0].legend()
axs[0].grid()

#Para o Exercício 3
axs[0].plot(x, U3(x), label = 'x**2 * (x - 1) * (x + 1) + x / 4 ')
axs[0].plot(traj_U3, [U3(xi) for xi in traj_U3], 'bo-', label= 'Trajetória')
axs[0].set_title('Exercício 3: Mútiplos Mínimos com Assimetria')
axs[0].legend()
axs[0].grid()

plt.tight_layout()
plt.show()
