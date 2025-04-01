import matplotlib.pyplot as plt
import numpy as np
import math as mt

def f(x): x**2 - 1

x = np.linspace(-3, 3, 10)

y = f(x)

plt.plot(x, y, label = "Mínimos da função")

plt.plot(0,f(0), 'ro' )

plt.show()  

