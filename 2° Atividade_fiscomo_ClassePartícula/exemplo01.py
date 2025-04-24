import numpy as np
import matplotlib.pyplot as plt

class Particula:
    def __init__(self, x_m, y_m, vx_m_s, vy_m_s, massa_kg, gravidade_g):
        self.x_m = x_m
        self.y_m = y_m
        self.vx_m_s = vx_m_s
        self.vy_m_s = vy_m_s
        self.massa_kg = massa_kg
        self.gravidade_g = gravidade_g

    def newton(self, fx, fy, dt):
        """
        Aplica a segunda lei de Newton para atualizar a velocidade e a posição da partícula.
        """
        self.vx_m_s += fx / self.massa_kg * dt
        self.vy_m_s += fy / self.massa_kg * dt
        self.x_m += self.vx_m_s * dt
        self.y_m += self.vy_m_s * dt

def plotar_trajetoria(particula, tempo_total, dt):
    tempos = np.arange(0, tempo_total, dt)
    x_vals = []
    y_vals = []

    for t in tempos:
        fx = 0  # Força na direção x (pode ser alterada conforme necessário)
        fy = -particula.massa_kg * particula.gravidade_g  # Força na direção y (gravidade)
        particula.newton(fx, fy, dt)
        x_vals.append(particula.x_m)
        y_vals.append(particula.y_m)

    plt.plot(x_vals, y_vals, label='Trajetória da partícula')
    plt.xlabel('Posição X (m)')
    plt.ylabel('Posição Y (m)')
    plt.title('Trajetória da Partícula')
    plt.grid()
    plt.legend()
    plt.show()

# Exemplo de uso
particula = Particula(x_m=0, y_m=0, vx_m_s=10, vy_m_s=10, massa_kg=1, gravidade_g=9.81)
plotar_trajetoria(particula, tempo_total=2, dt=0.1)