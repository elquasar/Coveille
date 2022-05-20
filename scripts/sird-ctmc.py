

import numpy as np
import matplotlib.pyplot as plt

"""
########################################################################
Ce script permet de simuler la formulation du problème SIRD en chaîne de 
Markov à temps continu. 
########################################################################
"""

# Paramètres de la simulation
N     = int(1000)         # population totale 
T     = 100         # temps de simulation en jour
nsim  = 3           # nombre de trajectoires simulées
# Paramètres du modèle
beta  = 0.3   # taux de transmission 
gamma = 1/10  # taux de guérison (10 jours de guérison)
nu    = 0.001 # taux de mortalité 
# Conditions initiales 
# taille du vecteur 2*T 
# initialiser M[0]

# solution pour nsim trajectoires
Z = np.zeros((nsim,2*N,4))          

for j in range(nsim):
    i = 0
    M = np.zeros((2*N,4))
    t = np.zeros(2*N)
    t[0] = 0
    X0 = np.array([N-10,10,0,0])
    M[0] = X0
    while M[i][1] > 0 and t[i] < T : 
        
        u1 = np.random.uniform(0,1)
        u2 = np.random.uniform(0,1)

        p1 = beta  * (M[i][0]*M[i][1]/N) #   beta*s*i/N
        p2 = gamma * M[i][1]             #   gamma * i
        p3 = nu    * M[i][1]             #   nu * i

        tot = p1+p2+p3
        transi1 = p1 / (tot)
        transi2 = p2 / (tot)
        transi3 = p3 / (tot)
        
        if 0 < u1 <= transi1: # nouvel infecté
            M[i+1][0] = M[i][0] - 1  
            M[i+1][1] = M[i][1] + 1
            M[i+1][2:3] = M[i][2:3]
        if transi1 < u1 <= transi1 + transi2: # nouveau guéri
            M[i+1][0] = M[i][0]
            M[i+1][1] = M[i][1] - 1   
            M[i+1][2] = M[i][2] + 1
            M[i+1][3] = M[i][3]
        if transi1 + transi2 < u1 <= transi1 + transi2 + transi3: # nouveau mort
            M[i+1][0] = M[i][0]
            M[i+1][1] = M[i][1] - 1   
            M[i+1][2] = M[i][2]
            M[i+1][3] = M[i][3] + 1
            
        # prochain evenemnt
        t[i+1] = t[i] - (np.log(u2) / tot)
        i = i+1
    Z[j] = M

plt.figure(figsize=(16,8))
plt.title("Simulation CTMC infectés")
plt.xlabel("Temps (en jours)")
plt.ylabel("Population")
for i in range(len(Z[:,0,0])): 
    plt.plot(t,Z[i,:,1],alpha=0.8,label= "Trajectoire {}".format(i+1))
plt.legend()
plt.show()




