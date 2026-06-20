import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve

# ----------------------PARAMÈTRES---------------------------

hbar = 1.0
m = 1.0

# Espace
N = 1500
x = np.linspace(-30, 30, N)
dx = x[1] - x[0]

# Temps
dt = 0.005
Nt = 2000

tab_temps = np.arange(Nt) * dt

# Paquet d'ondes
x0 = -15.0
sigma = 1.5                    # largeur du paquet
k0 = 5.0

# barriere potentiel
V0 = 13
a_barriere = 0.0
b_barriere = 1
epaisseur = b_barriere - a_barriere

# Ec paquet
E_moyenne = (hbar**2 * k0**2) / (2*m)       # de Broglie : p = hbar k
print(f"Énergie cinétique moyenne du paquet : E = {E_moyenne:.2f}")
print(f"Hauteur de la barrière : V0 = {V0}")
if V0 > E_moyenne:
    print("V0 > E : Effet tunnel observable")
else:
    print("V0 < E : Transmission classique")


# ----------------------Paquet Gauss initial--------------------------- (partie 2)

def paquet_gaussien(x, x0, sigma, k0):

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



# ----------------------Animation---------------------------

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(-30, 30)
ax.set_ylim(0, 1.15 * densite_proba.max())

ax.plot(x, V, color="gray", label="barrière de potentiel")
ax.axvspan(a_barriere, b_barriere, alpha=0.15)

ligne, = ax.plot([], [], lw=2, color="red", label=r"$|\psi(x,t)|^2$")

ax.set_xlabel("position : x")
ax.set_ylabel(r"densité de probabilité : $|\psi(x,t)|^2$")
ax.set_title("Paquet d'ondes Gaussien qui rencontre une barrière de potentiel")
ax.legend(loc="upper right")

def init():
    ligne.set_data([], [])
    return ligne,

def update(frame):
    ligne.set_data(x, densite_proba[frame])
    return ligne,

pas_affichage = 4                   # On n'affiche pas toutes les frames sinon trop lent
ani = animation.FuncAnimation(
    fig, update, frames=range(0, Nt, pas_affichage),
    init_func=init, interval=20
)

plt.show()
#ani.save("paquetGauss_barriere.gif", writer="pillow", fps=30)

#---------------------- question b ----------------------

def simu_libre():

    coef = hbar**2 / (2*m*dx**2)
    diag = 2*coef
    diag_supp_inf = -coef * np.ones(N - 1)

    E_libre = diags([diag_supp_inf, diag, diag_supp_inf], offsets=[-1, 0, 1])

    I = identity(N)
    A = (I + 1j*dt/(2*hbar) * E_libre)
    B = (I - 1j*dt/(2*hbar) * E_libre)

    # ----------------Evolution de psi-------------------- (partie 3)

    psi = psi0.copy()
    densite_proba = np.zeros((Nt, N))   # stock pour l'animation

    for n in range(Nt):
        densite_proba[n, :] = np.abs(psi)**2    #module carré pour la partie réelle
        second_membre = B.dot(psi)      #B psi n (dot produit matriciel)
        psi = spsolve(A, second_membre) #spsolve sert a determiner psi ()


    norme_finale = np.sum(np.abs(psi)**2) * dx # verif de la condition de normalisation si = 1, la particule est bien conservée tout le long

    print(f"Norme finale du paquet d'ondes : {norme_finale:.6f} ( = 1 ?)")  # la norme est stable jusqu'à 10^-12

    return densite_proba

def position_pic(densite_proba):
    max = np.argmax(densite_proba, axis=1)
    return x[max]

def temps(x_pic, borne):
    for i in range(len(x_pic)):
        if x_pic[i] >= borne:
            return tab_temps[i]
    return None

densite_libre = simu_libre()
mon_pic = position_pic(densite_libre)

temps_a = temps(mon_pic, a_barriere)
temps_b = temps(mon_pic, b_barriere)
tau0_num = temps_b - temps_a

print(f"tau0,num = {tau0_num:.4f}")
print(f"comparaison : a/v_g = {epaisseur/k0/m:.4f}")


#---------------------- question c ----------------------
def max_onde_transmis(densite_proba,limite):
    #On garde seulement le x de la fin de la barrière
    indice_fin_barriere = np.searchsorted(x, limite)
    position_max_transmis = np.full(Nt, np.nan)

    #On garde seulement la densité d'après la barrière pour trouver la position du pic de l'onde transmise
    for i in range(Nt):
        apres_barriere = densite_proba[i, indice_fin_barriere:]
        #On ignore tant que le paquet n'a pas franchit la barrière
        if(np.max(apres_barriere) > 1e-3):
            indice_max = np.argmax(apres_barriere)
            position_max_transmis[i] = x[indice_fin_barriere+indice_max]
        
    return position_max_transmis

def temps_barriere_atteint(x_pic, borne, tab_temps):
    #On cherche le temps où l'onde à bien atteint la barrière
    for i in range(len(x_pic)):
        #On verifie que la case du tableau est bien définie
        if(not np.isnan(x_pic[i]) and x_pic[i] > borne):
            return tab_temps[i]
    return None

x_surete = b_barriere + 10 #Le x où on est sûr que l’onde a bien passé la barrière

pic_transmis = max_onde_transmis(densite_proba, b_barriere)
temps_b_transmis = temps_barriere_atteint(pic_transmis, x_surete, tab_temps)
temps_a_transmis = temps(mon_pic, x_surete)

if(temps_b_transmis is not None):
    tau_t_num = tau0_num + temps_b_transmis - temps_a_transmis #On calcule le decalage avec et sans barrière
    print("\nTemps de franchissement de la barrère")
    print(f"tau_t,num = {tau_t_num:.4f} s")