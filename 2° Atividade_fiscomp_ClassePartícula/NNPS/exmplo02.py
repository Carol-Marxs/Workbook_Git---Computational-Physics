import numpy as np
import matplotlib.pyplot as plt

# esse gráfico é sobre a trajetória de uma partícula sob a ação da gravidade
#mas atarvessando a linha do zero 0

class Particula:
    def __init__(self, x0, y0, vx0, vy0, massa_kg, gravidade_g):
        self.massa = massa_kg
        self.g = gravidade_g
        self.t = 0  # tempo inicial

        # Listas para armazenar a trajetória
        self.x = [x0]
        self.y = [y0]
        self.vx = [vx0]
        self.vy = [vy0]
        self.tempo = [self.t]

    def newton(self, fx, fy, dt):
        # Acelerações
        ax = fx / self.massa
        ay = fy / self.massa

        # Últimos valores de posição e velocidade
        vx_n = self.vx[-1]
        vy_n = self.vy[-1]
        x_n = self.x[-1]
        y_n = self.y[-1]

        # Atualiza velocidades
        vx_new = vx_n + ax * dt
        vy_new = vy_n + ay * dt

        # Atualiza posições
        x_new = x_n + vx_new * dt
        y_new = y_n + vy_new * dt

        # Atualiza tempo
        self.t += dt

        # Armazena os novos valores
        self.vx.append(vx_new)
        self.vy.append(vy_new)
        self.x.append(x_new)
        self.y.append(y_new)
        self.tempo.append(self.t)

    def simular_trajetoria(self, dt):
        while self.y[-1] >= 0:
            fx = 0
            fy = -self.massa * self.g
            self.newton(fx, fy, dt)

    def plotar_trajetoria(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.x, self.y, label='Trajetória')
        plt.scatter(self.x[-1], self.y[-1], color='red', label='Impacto')
        plt.xlabel('Posição X (m)')
        plt.ylabel('Posição Y (m)')
        plt.title('Movimento da Partícula sob Gravidade')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# =============================
# Execução
# =============================

p = Particula(x0=0, y0=0, vx0=10, vy0=10, massa_kg=1, gravidade_g= 9.8)
p.simular_trajetoria(dt=0.1)
p.plotar_trajetoria()
