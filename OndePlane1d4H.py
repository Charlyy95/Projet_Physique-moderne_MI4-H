import numpy
from numpy import pi, exp, sqrt, real, imag, zeros, linspace, cos
import matplotlib.pyplot as plt

deux_pi = 2*pi

def PlaneWave(amp, k, omega, x, t):
    if(k<0):
        print("ca marche pas")
        exit()
    else:
        return amp * exp(1j*(k*x - omega*t))


t = 0
deltaK = 0.5
ko = pi
k = [ko - deltaK/2, ko, ko + deltaK/2]
x = linspace(-pi/deltaK, pi/deltaK, 1000)

omega = 10
amp = 2
nb = 3
superposition = 0

onde1 = PlaneWave(amp, ko, omega, x, t)
onde2 = PlaneWave(amp/2, ko-deltaK/2, omega, x, t)
onde3 = PlaneWave(amp/2, ko+deltaK/2, omega, x, t)

superposition = onde1 + onde2 + onde3
enveloppe = amp*(1+cos((deltaK * x)/2))

fig, ax = plt.subplots(figsize=(10,5))
fig.suptitle("Superposition d'ondes planes et l'enveloppe", fontsize=15)
ax.plot(x, real(onde1), label="Onde 1", color='blue')
ax.plot(x, real(onde2), label="Onde 2", color='green')
ax.plot(x, real(onde3), label="Onde 3", color='red')
ax.plot(x,numpy.real(superposition), label="Somme des 3 ondes planes", color='gray')
ax.plot(x, enveloppe, label="Enveloppe", color='blue', linestyle='--')
ax.plot(x, -enveloppe, color='blue', linestyle='--')
ax.set_xlabel("x")
ax.set_ylabel("Amplitude")
ax.grid(True)
ax.legend()
plt.savefig("Exercice 1.2 d.png",bbox_inches='tight')
plt.show()

