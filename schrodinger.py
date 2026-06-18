import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve

# ----------------------PARAMÈTRES---------------------------

hbar = 1.0
m = 1.0

# Espace
x_min, x_max = -30.0, 30.0
N = 1500                       # nombre de x 
x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]

# Temps
dt = 0.005
Nt = 2000                      # nombre total de pas de temps

# Paquet d'ondes
x0 = -15.0                     # position initiale paquet
sigma = 1.5                    # largeur du paquet
k0 = 5.0                       # vecteur d'onde

# barriere potentiel
V0 = 13
a_barriere, b_barriere = 0.0, 1.0
epaisseur = a_barriere - b_barriere

# Ec paquet
E_moyenne = (hbar**2 * k0**2) / (2*m)       # de Broglie : p = hbar k
print(f"Énergie cinétique moyenne du paquet : E = {E_moyenne:.2f}")
print(f"Hauteur de la barrière : V0 = {V0}")
if V0 > E_moyenne:
    print("V0 > E : Physique Classique : l'onde ne passe pas, Physique quantique : on observe l'effet tunnel.")
else:
    print("V0 < E : Physique Classique : une partie de l'onde va traverser")


# ----------------------Paquet Gauss initial--------------------------- (partie 2)

def paquet_gaussien(x, x0, sigma, k0):
    """Paquet d'ondes gaussien normalisé, de vecteur d'onde moyen k0."""
    morceau1 = (2*np.pi*sigma**2)**(-0.25)      #normalisation
    morceau2 = np.exp(-(x - x0)**2 / (4*sigma**2))
    morceau3 = np.exp(1j*k0*x)
    return morceau1 * morceau2 * morceau3

psi0 = paquet_gaussien(x, x0, sigma, k0)

# ----------------------Potentiel et Energie---------------------------
'''aide de l'ia Claude pour cette partie pour la construction de la matrice E'''
V = np.where((x >= a_barriere) & (x <= b_barriere), V0, 0.0)    #on definit un tableau qui rpz le potentiel

coef = hbar**2 / (2*m*dx**2)
diag = 2*coef + V            #psi i
diag_supp_inf = -coef * np.ones(N - 1)     #psi i+1 et psi i-1

E = diags([diag_supp_inf, diag, diag_supp_inf], offsets=[-1, 0, 1])



I = identity(N)
A = (I + 1j*dt/(2*hbar) * E)        #Crank-Nicolson : (I + i dt/(2 hbar) H) psi^(n+1) = (I - i dt/(2 hbar) H) psi^(n)
B = (I - 1j*dt/(2*hbar) * E)                                           # diag_supp_inf E                    diag E



# ----------------------Evolution de psi--------------------------- (partie 3)

psi = psi0.copy()
densite_proba = np.zeros((Nt, N))   # stock pour l'animation

for n in range(Nt):
    densite_proba[n, :] = np.abs(psi)**2    #module carré pour la partie réelle
    second_membre = B.dot(psi)      #B psi n (dot produit matriciel)
    psi = spsolve(A, second_membre) #spsolve sert a determiner psi ()


norme_finale = np.sum(np.abs(psi)**2) * dx # verif de la condition de normalisation si = 1, la particule est bien conservée tout le long

print(f"Norme finale du paquet d'ondes : {norme_finale:.6f} ( = 1 ?)")  # la norme est stable jusqu'à 10^-12



