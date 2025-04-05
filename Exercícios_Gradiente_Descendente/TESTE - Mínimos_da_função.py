import matplotlib.pyplot as plt
import numpy as np
import math as mt

#Exercício_1

def U(x):
    return x**2 - 1

x = np.linspace(-3, 3, 100)

y = U(x)

plt.plot(x, y, label = "Mínimos da função")
plt.plot(0, U(0), 'ro' )
plt.text(0, U(0), f' Mínimo (0, -1)')


#EXERCÍCIO_2

def U(X):
    return X**2 * (x - 1) * (x + 1)

x = np.linspace(-3, 3, 100)
y = U(x)

plt.plot(x, y, label = 'Mínimo da função')
plt.plot(0, U(0), 'ro-')
plt.tex(0, U(0), f'Mínimo(0,-1)')
plt.show()