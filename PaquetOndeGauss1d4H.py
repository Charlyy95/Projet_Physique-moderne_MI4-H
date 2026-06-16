import numpy
from numpy import pi, exp, sqrt, real, imag, zeros, linspace, cos, pow
import matplotlib.pyplot as plt

pi2 = 2* pi
m = 9.10e-31
hbar = 1.054e-34

def GaussWP(ko, a, x, t):
    morceau1 = pow(1/(8*(pi**3)),1/4)
    morceau2 = sqrt((4*pi*m*a)/(m*(a**2) + 2*1j*hbar*t))
    morceau3 = exp((m/4)*((((a**2)*ko + 2*1j*x)**2)/(m*(a**2) + 2*1j*hbar*t)) - ((a**2)*(ko**2))/4)
    return morceau1*morceau2*morceau3

if __name__ == "__main__":      #empeche le reste du code de s'éxécuter quand on l'appelle depuis un autre fichier
    ko = 5
    a = 1e-10
    t = 0
    x = linspace(-10e-10, 10e-10, 1000)
    onde = GaussWP(ko,a,x,t)

    fig, ax = plt.subplots(figsize=(10,5))
    fig.suptitle("Paquet d'ondes gaussien test à t=0 avec a valant la largeur d'un atome", fontsize=15)
    ax.plot(x, real(onde), label="Partie réelle", color='blue')
    ax.plot(x, imag(onde), label="Partie imaginaire", color='red')
    ax.set_xlabel("x")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)
    plt.savefig("Exercice 2.2 avec t=0 bonne echelle.png",bbox_inches='tight')
    plt.show()
    plt.close()