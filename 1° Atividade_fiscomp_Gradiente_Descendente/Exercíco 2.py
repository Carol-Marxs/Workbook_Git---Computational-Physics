                        #EXERCÍCIO 2

import numpy as np
import matplotlib.pyplot as plt

#funções

def U2(x):
    return x**2 * (x - 1) * (x + 1)

def grad_U2(x):
    return 4 * x**3 - 2 * x

#parametros do gradiente descendente

def gradiente_descendente(f, grad_f, x0, alpha = 0.1, epsilon = 0.02, max_iter = 1000):
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

#parametros das partículas

traj1 = gradiente_descendente(U2, grad_U2, x0 = -2, alpha=0.12)
traj2 = gradiente_descendente(U2, grad_U2, x0 = 2, alpha=0.12)
traj3 = gradiente_descendente(U2, grad_U2, x0= 1.7, alpha=0.12) #teste de visualização



# plot do Gráfico
x = np.linspace(-2, 2 , 500)
plt.plot(x, U2(x), label= 'x**2 * (x - 1) * (x + 1)', color='blue')
plt.plot(traj1, [U2(xi) for xi in traj1], 'yo-', label= 'Trajetória da partícula 1')
plt.plot(traj2, [U2(xi) for xi in traj2], 'go-', label= 'Trajetória da partícula 2')
plt.plot(traj3, [U2(xi) for xi in traj3], 'ro-', label= 'Trajetória da partícula 3')
plt.title('Gradiente Descendente - Exercício 2')
plt.xlabel('x')
plt.ylabel(U2(x))
plt.grid()
plt.legend()
plt.show()


 # $$O que acontece? O que você pode concluir sobre a escolha da taxa de aprendizado α?

# Como diz no anuciado do exercício, o gráfico dessa função exibe dois minímos globais,
# plotei simutaneamente os dois mínimos onde a escolha de alpha muda o passo para chegar nos mínimos. 
# Os valores de alpha afeta fortemente a convergência, se muito pequeno a partícula converge lentamente, mas se grande, a partícula oscila até divergir. 
# Mas há um meio termo onde a particula pode chegar rapidamente e com estabilidade. 