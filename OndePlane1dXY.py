import numpy
from numpy import pi, exp, sqrt, real, imag, zeros, linspace
import matplotlib.pyplot as plt

deux_pi = 2*pi

def PlaneWave(amp, k, omega, x, t):
    if(k<0):
        print("ca marche pas")
        exit
    else:
        return amp * exp(1j*(k*x - omega*t))


x = linspace(0,100,1000)
t = 0
k = 0.3
omega = 10
amp = 0.1
nb = 3

onde = PlaneWave(amp,k,omega,x,t)

fig, ax = plt.subplots(1,1,figsize=(10,5))
fig.suptitle("Test Onde Plane", fontsize=14)
ax.plot(x,numpy.real(onde))
ax.grid('True')
plt.savefig("test.png",bbox_inches='tight')
plt.show()

