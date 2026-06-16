import numpy
from numpy import pi, exp, sqrt, real, imag, zeros, linspace, cos, pow
import math
import matplotlib.pyplot as plt

from PaquetOndeGauss1d4H import GaussWP

pi2 = 2* pi
m = 1
#unité réduite : hbar = m = 1 sinon beaucoup trop trop long 
hbar = 1
#--------------------3.1--------------------


coord_x = numpy.linspace(0,100,1000)
coord_x2 = []


for i in range(len(coord_x)):
    coord_x2.append(pow(coord_x[i], 2))
    



pas = 0

for i in range(1):
    print(f"Pas: {coord_x[i+1]- coord_x[i]}")
    pas = coord_x[i+1]- coord_x[i]


def calculer_derive(coord_x , coord_x2):

    derive_x = []

    for i in range(len(coord_x)-1):

        derive_x.append((coord_x2[i+1]-coord_x2[i])/(coord_x[i+1]-coord_x[i]))
    
    return derive_x

def calculer_derive2nde(coord_x , coord_x2):

    derive2_x = []
    diff = coord_x[1] - coord_x[0]

    for i in range(1, len(coord_x)-1):

        derive2_x.append((coord_x2[i+1]- 2*coord_x2[i] + coord_x2[i-1])/(diff**2))          #fi+1 - 2fi + fi-1 / dx**2
    
    return derive2_x

def double(x, h):
    derive_theorique = []
    for i in range(len(x)):
        derive_theorique.append(2*x[i] + h)
    return derive_theorique

def marge_erreur(derive_x , derive_theorique):

    
    pourcentage_erreur = []
    for i in range(len(derive_x)):
        if(derive_theorique[i] != 0):
            pourcentage_erreur.append((abs((derive_x[i]-derive_theorique[i]))/(derive_theorique[i]))*100)
        
    return pourcentage_erreur
    

derive_x = calculer_derive(coord_x, coord_x2)

double_x = double(coord_x, pas)
'''
for i in range(20):
    print(f"x:{coord_x[i]} | x²:{coord_x2[i]} | 2x: {derive_x[i]} | Derivée Theorique: {double(coord_x, pas)[i]} | Pourcentage Erreur: {marge_erreur(derive_x, double_x)[i]}%")



derive2_x = calculer_derive2nde(coord_x, coord_x2)
for i in range(10):
    print(f"x:{coord_x[i+1]:.3f} | d²x²/dx² = {derive2_x[i]:.6f} | Théorique: 2.000000")
'''
#--------------------3.2--------------------

nx = 100   # nbre points espace
nt = 500    # nbre pas temps

psi = numpy.empty((nx, nt), dtype=complex)  #rempli tout le vide

ko = 2.0
a  = 1.0
x = linspace(-10, 10, nx)

# Q3 : intervalles
t  = numpy.linspace(0, 2, nt)
dx = x[1] - x[0]
dt = t[1] - t[0]


psi[:, 0] = numpy.exp(-x**2 / (2*a**2)) * numpy.exp(1j * ko * x)    # colonne 0 = paquet initial
psi[:, 0] /= numpy.sqrt(numpy.sum(numpy.abs(psi[:, 0])**2) * dx)    # normalisation  

#Q4 : evolution de psi

for n in range(nt - 1):
    for i in range(1, nx - 1):
        d2psi = (psi[i+1, n] - 2*psi[i, n] + psi[i-1, n]) / dx**2       #derivee seconde x
        psi[i, n+1] = psi[i, n] + (1j * hbar * dt / (2 * m)) * d2psi
    psi[0, n+1]  = 0    #conditions aux bords
    psi[-1, n+1] = 0

    # Vérification à chaque pas
    norme = numpy.sum(numpy.abs(psi[:, n+1])**2) * dx
    if norme > 100 or numpy.isnan(norme):
        print(f"Explosion à n={n}, norme={norme:.4e}")
        break

'''
fig, ax = plt.subplots(figsize=(10,5))
fig.suptitle("Évolution du paquet d'ondes | Particule libre (V₀=0)", fontsize=15)

instants = numpy.linspace(0, nt-1, 4, dtype=int)
for idx in instants:
    ax.plot(x, numpy.abs(psi[:, idx])**2, label=f"t={t[idx]:.2e} s")
ax.set_xlabel("x")
ax.set_ylabel("Densité de probabilité")
ax.legend()
ax.grid(True)
plt.savefig("Exercice 3.2.png",bbox_inches='tight')
plt.show()
plt.close()
'''


# Q5 : comparaison avec GaussWP
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Q5 — Comparaison Numérique vs Analytique (GaussWP)", fontsize=15)

# Gauche : numérique
instants = numpy.linspace(0, nt-1, 4, dtype=int)
for idx in instants:
    axes[0].plot(x, numpy.abs(psi[:, idx])**2, label=f"t={t[idx]:.2f}")
axes[0].set_title("Numérique (unités réduites, V₀=0)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("|ψ|²")
axes[0].legend()
axes[0].grid(True)

# Droite : analytique GaussWP (unités SI)
x_SI = numpy.linspace(-10e-10, 10e-10, nx)
k0_SI = 1.73e11

for t_val in [0, 1e-17, 2e-17, 3e-17]:
    onde = GaussWP(k0_SI, 1e-10, x_SI, t_val)
    prob = numpy.abs(onde)**2
    axes[1].plot(x_SI * 1e10, prob, label=f"t={t_val:.1e} s")

axes[1].set_title("Analytique GaussWP (unités SI)")
axes[1].set_xlabel("x (×0.1 nm)")
axes[1].set_ylabel("|ψ|²")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("Exercice 3.2_Q5_comparaison.png", bbox_inches='tight')
plt.show()
plt.close()

