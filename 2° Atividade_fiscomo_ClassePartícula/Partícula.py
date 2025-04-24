import math
import numpy as np
import matplotlib.pyplot as plt

# Atributos:

# x: coordenada x da partícula
# y: coordenada y da partícula
# vx: velocidade na direção x
# vy: velocidade na direção y
# massa: massa da partícula

class Particula:
    def __init__(self, x, y, vx, vy, massa): #construção da classe que inializa os atributos da particula
        self.x = x
        self.y = y  
        self.vx = vx
        self.vy = vy
        self.massa = massa

def newton(self, fx, fy, dt): # aplica a segunda lei de Newton para atualizar a velocidade e a posição da partícula com base nas forças aplicadas:
    # fx: força na direção x
    # fy: força na direção y
    # dt: intervalo de tempo
    # Atualiza a velocidade e a posição da partícula usando as forças aplicadas
    self.vx += fx / self.massa * dt
    self.vy += fy / self.massa * dt
    self.x += self.vx * dt
    self.y += self.vy * dt

#plotando a trajetória da partícula
def plotar_trajetoria(particula, tempo_total, dt):
    tempos = np.arange(0, tempo_total, dt)
    x_vals = []
    y_vals = []

    for t in tempos:
        fx = 0  # Força na direção x (pode ser alterada conforme necessário)
        fy = -particula.massa * particula.gravidade_g  # Força na direção y (gravidade)
        particula.newton(fx, fy, dt)
        x_vals.append(particula.x)
        y_vals.append(particula.y)

    plt.plot(x_vals, y_vals, label='Trajetória da partícula')
    plt.xlabel('Posição X (m)')
    plt.ylabel('Posição Y (m)')
    plt.title('Trajetória da Partícula')
    plt.grid()
    plt.legend()
    plt.show()

