import numpy
import math



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

for i in range(20):
    print(f"x:{coord_x[i]} | x²:{coord_x2[i]} | 2x: {derive_x[i]} | Derivée Theorique: {double(coord_x, pas)[i]} | Pourcentage Erreur: {marge_erreur(derive_x, double_x)[i]}%")



