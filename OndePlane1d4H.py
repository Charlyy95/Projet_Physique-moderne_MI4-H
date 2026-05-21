import numpy
from numpy import pi, exp, sqrt, real, imag, zeros, linspace
import matplotlib.pyplot as plt

deux_pi = 2*pi

def PlaneWave(amp, k, omega, x, t):
    if(k<0):
        print("ca marche pas")
        exit()
    else:
        return amp * exp(1j*(k*x - omega*t))


x = linspace(0,100,1000)
t = 0
deltaK = 0.1
ko = pi
k = [ko - deltaK/2, ko, ko + deltaK/2]

omega = 10
amp = ko/2
nb = 3
superpostion = 0

for i in range(nb):
    ondePlane = PlaneWave(amp,k[i],omega,x,t)
    superpostion += ondePlane

fig, ax = plt.subplots(figsize=(10,5))
fig.suptitle("Exercice 1.2 d", fontsize=14)
ax.plot(x,numpy.real(superpostion))
ax.grid(True)
plt.savefig("Exercice 1.2 d.png",bbox_inches='tight')
plt.show()

