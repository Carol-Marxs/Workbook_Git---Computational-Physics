import numpy as np #importar funções ou constantes específicas do módulo 
import matplotlib.pyplot as plt #criar gráficos
import math #biblioteca matemática

def f(x):
    return x**2-1

x = np.linspace(-3,3,10)

y = f(x)

plt.plot(x, y, label = "Mínimo da Função")

plt.plot(0, f(0), 'ro')
plt.text(0, f(0), f' Mínimo (0, -1)')
plt.show()