#              Exercício 1
import numpy as np
import matplotlib.pyplot as plt

# Função U(x) = x² - 1 e sua derivada
def U1(x):
    return x**2 - 1

def grad_U1(x):
    return 2 * x

# Algoritmo de gradiente descendente
def gradiente_descendente(f, grad_f, x0, alpha=0.1, epsilon=0.1, max_iter=1000):
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
epsilon = 0.1
traj = gradiente_descendente(U1, grad_U1, x0, alpha)

# Gráfico
x = np.linspace(-6, 6, 1000)
plt.plot(x, U1(x), label='U1(x) = x² - 1', color='blue')
plt.plot(traj, [U1(xi) for xi in traj], 'ro-', label='Trajetória da partícula')
plt.title('Gradiente Descendente - Exercício 1')
plt.xlabel('x')
plt.ylabel('U1(x)')
plt.grid()
plt.legend()
plt.show()

#                                                           DISCUSSÕES DO EXERCÍCIO 1 

# Nas configurações iniciais do exercício 1, o gráfico apresenta um caminho até chegar no mínomo da função rente a forma de linha da gráfico, ponto a ponto.
# Minha primeiro modificação foi fixar epsilon = 0.01 e variar o alpha em 0.3 e 0.5. 
# Em alpha = 0.3, a distância dos pontos é maior até chegar no mínimo da função, diminuindo assim, a quantidade de pontos, quando comparado ao alpha = 0.1.
# Em alpha = 0.5, a distândia dos pontos ainda maior, o mínimo da função é encontrado, mas há apenas dois pontos, o incial de x0 = 5 e xf = 0.
# Concluindo assim, que alpha controla o tamanho do passo que o ponto dá na direção oposta ao gradiente da função que está sendo minimizada.
# Quando alpha é pequeno a convergência é mais lenta, mas mais segura e precisa. Alpha grande os passos grandes pode convergir mais rápido,
# mas também pode saltar o mínimo, podendo assim, perder informação.

# Agora fixando alpha = 0.1 e variando epsilon em 0.1 e 0.001.
# Epsilon = 0.1 é longe de zero, quando comparado ao 0.001 ou 0.01. 
# Com isso, epsilon define quando o algoritmo deve parar, quando consideramos que estamos suficientemente próximos do mínimo.
