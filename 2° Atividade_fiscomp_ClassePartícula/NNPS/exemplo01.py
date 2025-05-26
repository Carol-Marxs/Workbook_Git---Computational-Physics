import numpy as np
import matplotlib.pyplot as plt

# essa classe representa uma partícula em movimento sob a influência da gravidade
# e permite calcular sua trajetória em um gráfico 2D. Com a partícula parando antes do zero
# e não atravessando a linha do zero 0
class Particula:
    def __init__(self, x_m, y_m, vx_m_s, vy_m_s, massa_kg, gravidade_g):
        self.x_m = x_m
        self.y_m = y_m
        self.vx_m_s = vx_m_s
        self.vy_m_s = vy_m_s
        self.massa_kg = massa_kg
        self.gravidade_g = gravidade_g

    def newton(self, fx, fy, dt):
        # Segunda Lei de Newton
        ax = fx / self.massa_kg
        ay = fy / self.massa_kg
        self.vx_m_s += ax * dt
        self.vy_m_s += ay * dt
        self.x_m += self.vx_m_s * dt
        self.y_m += self.vy_m_s * dt

def plotar_trajetoria(particula, dt):
    x_vals = []
    y_vals = []
    vx_vals = []
    vy_vals = []
    tempos = []

    t = 0
    while particula.y_m >= 0:
        x_vals.append(particula.x_m)
        y_vals.append(particula.y_m)
        vx_vals.append(particula.vx_m_s)
        vy_vals.append(particula.vy_m_s)
        tempos.append(t)

        fx = 0
        fy = -particula.massa_kg * particula.gravidade_g
        particula.newton(fx, fy, dt)
        t += dt

    # Gráfico da trajetória (x vs y)
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, 'b-', label='Trajetória')
    plt.scatter(x_vals[-1], y_vals[-1], color='red', label='Impacto')
    plt.xlabel('Posição X (m)')
    plt.ylabel('Posição Y (m)')
    plt.title('Trajetória da Partícula sob Gravidade')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return tempos, x_vals, y_vals, vx_vals, vy_vals

# Parâmetros iniciais
particula = Particula(x_m=0, y_m=0, vx_m_s=10, vy_m_s=10, massa_kg=1, gravidade_g=9.8)
tempos, x_vals, y_vals, vx_vals, vy_vals = plotar_trajetoria(particula, dt=0.1)
