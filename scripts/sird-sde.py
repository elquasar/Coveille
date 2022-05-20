
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
N     = 1000         # population totale 
T     = 100         # temps de simulation en jour
dt    = 0.1        # pas de temps de la simulation
n     = int(T/dt)   # nombre d'itérations
nsim  = 10           # nombre de trajectoires simulées
t = np.linspace(0,T,n)
# Paramètres du modèle
beta  = 0.8      # taux de transmission 
gamma = 0.3       # taux de guérison (10 jours de guérison)
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

borne = np.zeros((n,len(X0)-1,2))
sigma = 1.7

for _ in range(nsim):

    # Matrice Nbre de point*3,contenant les solutions de X a l instant t
    M = np.zeros((n, dim))
    # Vecteur solution pour chaque simulation
    # t = 0 initialisation du vecteur
    M[0] = X0

    for i in range(n-1):

        # Euler-Maruyama
        m = drift(M[i][0:-1],N,beta,gamma,nu)
        v = diffusion(M[i][0:-1],N,beta,gamma,nu)
        M[i+1][0:-1] = M[i][0:-1] + m*dt + np.sqrt(dt)*np.dot(
            v, eta(dim-1).T).reshape(1, dim-1)[0]
        M[i+1][-1] = N - np.sum(M[i+1][0:-1])
        borne[i+1,:,0] = M[i+1][0:-1] - sigma * np.diag(v)
        borne[i+1,:,1] = M[i+1][0:-1] + sigma * np.diag(v)
    solution[_] = M


print(borne)




plt.figure(figsize=(16,8))
plt.title("Simulation SDE infectés")
plt.xlabel("Temps (en jours)")
plt.ylabel("Population")
for i in range(len(solution[:,0,0])): 
    plt.plot(t,solution[i,:,1],alpha=0.8,label= "Trajectoire {}".format(i+1))
plt.fill_between(t, borne[:,1,0], borne[:,1,1],color='k',alpha=0.1,label="Intervalle valeurs possibles")
plt.legend()
plt.show()


def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density



# chaque point de la distribution représente une loi normale. 
# on peut dessiner sur la courbe les densités à chaque itérations
solution = solution[0,:,:]
x = np.linspace(0,20,N)
density = np.zeros((n,len(X0)-1,len(x)))
for i in range(n):
    # valeur possibles de Xk+1 à 1sigma
    mean = drift(solution[i,0:-1], N, beta, gamma, nu)
    var = np.diag(diffusion(solution[i,0:-1],N, beta, gamma, nu))
    
    for j in range(len(X0)-1):

        density[i,j,:] = normal_dist(x,mean[j],var[j])

plt.clf()
for k in range(n):
    if k%12==0:
        plt.plot(x, density[k,1,:])
plt.show()