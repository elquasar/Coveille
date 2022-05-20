"""
    Ce script permet la simulation des modèles d'épidémie de type SIR, les simulations incluent les modèles stochastiques et déterministes
"""


import numpy as np
import matplotlib.pyplot as plt
import datetime

from models.SIRD import SIRD
from models.SIRDV import SIRDV

from models.tools.constant import DATE_FRANCE,N
from models.tools.functions import give_compartiment
plt.rcParams["figure.figsize"] = (20,8)



if __name__ == '__main__':

    #################################
    #  Paramètres de la simulation  #

    T = 150                         # Temps de simulation en jour
    dt = 0.1                        # Pas de temps de la simulation
    nsim = 50                       # Nombre de trajectoires simulées pour la simulation stochastique
    # /!\ NE PAS MODIFIER /!\
    n = int(T/dt)                   # Nombre de points simulés
    t = np.linspace(0,T,int(T/dt))  # Vecteur de temps
    #################################

    

    #################################
    ##   Instanciation d'un modèle ##
            # MODÈLE SIRDV #
    sirdv = SIRDV()
    # Paramètres du modèle 
    sirdv.setParam({
    'N': 1000,    # Population totale
    'beta':0.51,  # Taux de transmission   0 < beta  < 1   
    'gamma':1/7,  # Taux de guérison (1/j) 0 < gamma < 1
    'nu':0.001,   # Taux de létalité       0 < nu    < 1
    'phi':0.036,    # Taux de vaccination    0 < phi   < 1
    'sigma':1-0.8   # 1 - Taux d'efficatité  0 < sigma < 1
    })
    # Vecteur de conditions initiales (S0,I0,R0,D0,V0)
    X0 = np.array([1000-1,1,0,0,0]) 
    ##################################
    
    ##################################
    #      Simulation du modèle      #
    sirdv.simulate(
        X0,                     # Conditions initiales
        t,                      # Vecteur de temps
        dt,                     # Pas de temps
        nsim,                   # Nombre de simulations
       type_simulation="stoch"    # Type de simulation parmi (all,deter,stoch,dtmc,ctmc)
        )
    ##################################

    ##################################
    # Affichage des courbes simulées #

    # Affichage compartiment S (composante 0)
    sirdv.mplot(0,stoch=True, det=False, conf=False, alpha=0.1, dtmc=False)
    # Affichage compartiment I (composante 1)
    sirdv.mplot(1,stoch=True, det=False, conf=False, alpha=0.1, dtmc=False)
    # Affichage compartiment R (composante 2)
    #sirdv.mplot(2,stoch=True, det=False, conf=False, alpha=0.1, dtmc=False)
    # Affichage compartiment D (composante 3)
    sirdv.mplot(3,stoch=True, det=False, conf=False, alpha=0.1, dtmc=False)
    # Affichage compartiment V (composante 4)
    sirdv.mplot(4,stoch=True, det=False, conf=False, alpha=0.1, dtmc=False)
    # /i\ Commenter pour retirer une courbe /i\
    # Utiliser plt.plot() pour afficher la figure
    plt.show()
    ##################################

