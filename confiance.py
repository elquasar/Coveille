
"""
########################################################################
Ce script permet de simuler la formulation du problème SIRD SDE
########################################################################
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import lower
from scipy.linalg import cholesky
# Reproductibilité de la simulation



# Paramètres de la simulation
N     = 10000         # population totale 
T     = 40         # temps de simulation en jour
dt    = 0.01        # pas de temps de la simulation
n     = int(T/dt)   # nombre d'itérations
nsim  = 100           # nombre de trajectoires simulées
t = np.linspace(0,T,n)
# Paramètres du modèle
beta  = 0.8       # taux de transmission 
gamma = 0.2       # taux de guérison (10 jours de guérison)
nu    = 0.02      # taux de mortalité 
# Conditions initiales  
X0 = np.array([N-10,10,0,0]) 

dim = len(X0)
n = len(t)
solution = np.zeros((nsim, n, dim))
def eta(size): return np.random.normal(0, 1, size)[np.newaxis]
def diffusion(X,N,beta,gamma,nu):

    if X[1] < 0:
        return np.zeros((len(X),len(X)))
    else :
        a = np.array([
            [beta*X[0]*X[1]/N, -beta*X[0]*X[1]/N, 0],
            [-beta*X[0]*X[1]/N, (beta*X[0]*X[1]/N) + gamma *
                X[1] + nu*X[1], -gamma*X[1]],
            [0, -gamma*X[1], gamma*X[1]]
        ])
        return cholesky(a,lower=True)
def drift(X,N,beta,gamma,nu):
    if X[1] < 0:
        return np.zeros(len(X))
    else :
        return np.array([
            -beta*X[0]*X[1]/N,
            (beta*X[0]*X[1]/N)-gamma*X[1] - nu*X[1],
            gamma*X[1]
        ])

for _ in range(nsim):

    # Matrice Nbre de point*3,contenant les solutions de X a l instant t
    M = np.zeros((n, dim))
    # Vecteur solution pour chaque simulation
    # t = 0 initialisation du vecteur
    M[0] = X0

    for i in range(n-1):

        # Euler-Maruyama

        M[i+1][0:-1] = M[i][0:-1] + drift(M[i][0:-1],N,beta,gamma,nu)*dt + np.sqrt(dt)*np.dot(
            diffusion(M[i][0:-1],N,beta,gamma,nu), eta(dim-1).T).reshape(1, dim-1)[0]
        M[i+1][-1] = N - np.sum(M[i+1][0:-1])
    solution[_] = M

plt.figure(figsize=(16,8))
plt.title("Simulation SDE infectés")
plt.xlabel("Temps (en jours)")
plt.ylabel("Population")
for i in range(len(solution[:,0,0])): 
    plt.plot(t,solution[i,:,1],alpha=0.8,label= "Trajectoire {}".format(i+1))
plt.legend()
plt.show()