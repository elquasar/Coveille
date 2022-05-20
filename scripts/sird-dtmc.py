import numpy as np
import matplotlib.pyplot as plt

"""
########################################################################
Ce script permet de simuler la formulation du problème SIRD en chaîne de 
Markov à temps discret. 
########################################################################
"""
# Reproductibilité de la simulation

np.random.seed(1000)


# Paramètres de la simulation
N     = 100         # population totale 
T     = 100         # temps de simulation en jour
dt    = 0.01        # pas de temps de la simulation
n     = int(T/dt)   # nombre d'itérations
nsim  = 3           # nombre de trajectoires simulées
# Paramètres du modèle
beta  = 0.3   *dt     # taux de transmission 
gamma = 1/10  *dt     # taux de guérison (10 jours de guérison)
nu    = 0.001 *dt     # taux de mortalité 
# Conditions initiales  

M = np.zeros((n,4))               # matrice solution du système
                                  # à chaque itérations
X0 = np.array([95,5,0,0])         # S = 90, I = 2, R = D = 0
M[0,:] = X0                       # Initialisation de la matrice solution
Z = np.zeros((nsim,n,4))          # matrice solution du système pour 
                                  # chaque trajectoires
# Itérations 
for j in range(nsim): 
    for i in range(0,n-1) : 
        
        S = M[i][0]
        I = M[i][1]
        R = M[i][2]
        D = M[i][3]
        # calcul de p1, p2, p3 et p4
        p1 = beta *(S*I/N)  # delta_t * beta*s*i/N
        p2 = gamma * I      # delta_t * gamma * i
        p3 = nu*I           # delta_t * nu * i
        p4 = 1 - (p1 + p2 + p3)
                    
        
        
        # tirage aléatoire loi uniforme 
        
        u = np.random.uniform(0,1)
        
        # tirage du prochain état 
        if 0 < u <= p1 :
            M[i+1,:] = np.array([S-1,I+1,R,D])

        if p1 < u <= p1 + p2 :
            M[i+1,:] = np.array([S, I - 1, R + 1, D])

        if p1 + p2 < u <= p1 + p2 + p3 :
            M[i+1,:] = np.array([S, I - 1, R, D + 1])

        if p1 + p2 + p3 < u <= 1 :
            M[i+1,:] = M[i,:]
    # ajout de la simulation au vecteur Z
    Z[j] = M
    
    
    
# affichage du graphe

t = np.linspace(0,T,n)    # grille de temps

plt.figure(figsize=(16,8))
plt.title("Simulation DTMC infectés")
plt.xlabel("Temps (en jours)")
plt.ylabel("Population")
for i in range(len(Z[:,0,0])): 
    plt.plot(t,Z[i,:,1],alpha=0.8,label= "Trajectoire {}".format(i+1))
plt.legend()
plt.show()




# Maximum likelihood estimator. 

# selectionner une trajectoire

# beta  = 0.3   *dt     # taux de transmission 
# gamma = 1/10  *dt     # taux de guérison (10 jours de guérison)
# nu    = 0.001 *dt     # taux de mortalité 


X = Z[1:,:]


