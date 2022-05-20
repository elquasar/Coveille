import numpy as np
import matplotlib.pyplot as plt


from models.SIRD import SIRD
from models.SIRDV import SIRDV

plt.rcParams["figure.figsize"] = (20,8)

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.optimize import basinhopping




plt.rcParams["figure.figsize"] = (20,10)
def drift(X,params):
    beta,gamma,nu = params
    if X[1] <= 0:
        return np.zeros(len(X))
    else :
        return np.array([
                -beta*X[0]*X[1]/N,
                (beta*X[0]*X[1]/N)-gamma*X[1] - nu*X[1],
                gamma*X[1]
            ])

def diffusion(X,params):
    beta,gamma,nu = params
    if X[1] <= 0:
        return np.zeros((len(X),len(X)))

    a = np.array([
        [beta*X[0]*X[1]/N, -beta*X[0]*X[1]/N, 0],
        [-beta*X[0]*X[1]/N, (beta*X[0]*X[1]/N) + gamma *
            X[1] + nu*X[1], -gamma*X[1]],
        [0, -gamma*X[1], gamma*X[1]]
    ])
    return a



def log_likelihood(params,data):
    if np.isnan(params).any():
        params = [0.5,0.3,0.01]
    sum = 0
    for i in range(0,len(data[:,0])-1):
        mean = data[i,:] + dt*drift(data[i,:],params)
        var = dt*diffusion(data[i,:],params)
        if np.all(var==0):
            sum+= 0
        else :
            sum += np.log(multivariate_normal.pdf(data[i+1,:],mean=mean,cov=var))

    # print("####################")
    # print("parameter :",params)
    # print("####################")

    return -sum

if __name__ == '__main__':

    #################################
    #  Paramètres de la simulation  #

    T = 40                         # Temps de simulation en jour
    dt = 0.1                        # Pas de temps de la simulation
    nsim = 1                     # Nombre de trajectoires simulées pour la simulation stochastique
    # /!\ NE PAS MODIFIER /!\
    n = int(T/dt)                   # Nombre de points simulés
    t = np.linspace(0,T,int(T/dt))  # Vecteur de temps
    #################################
    N = 1e3
    

    #################################
    ##   Instanciation d'un modèle ##
            # MODÈLE SIRD #
    sird = SIRD()
    # Paramètres du modèle 
    sird.setParam({
    'N': N,   # Population totale
    'beta':0.9, # Taux de transmission   0 < beta  < 1   
    'gamma':0.9, # Taux de guérison (1/j) 0 < gamma < 1
    'nu':0.0001   # Taux de létalité       0 < nu    < 1
    })
    # Vecteur de conditions initiales (S0,I0,R0,D0)
    X0 = np.array([N-10,10,0,0]) 



    ##################################
    ##################################
    #      Simulation du modèle      #
    sird.simulate(
        X0,   # Conditions initiales
        t,    # Vecteur de temps
        dt,   # Pas de temps
        nsim, # Nombre de simulations
        type_simulation="deterministe"
        )


    sird.aplot(stoch=True,alpha=0.5)
    plt.show()
    #################################
    #################################
         Estimation du modèle      #
    params = []
    datas = sird.sol_stoch


    data = datas[0,:,:]
    data = data[:,0:-1]
    # nettoyage data < 0
    data[data < 0] = 0

    chunks = np.split(data,20)
    params = []
    params0 = [0.89,0.3,0.01] 


    bounds = ((0.1,0.99),(0.1,0.99),(0.01,0.99))
    # 'L-BFGS-B'






    for chunk in chunks :
        minimizer_kwargs = {
        "args": (chunk),
        "bounds" : bounds,
        "method": 'L-BFGS-B'
        }
        res =  basinhopping(
            log_likelihood,
            x0=params0,
            minimizer_kwargs=minimizer_kwargs,
        )
        params.append(res.x)
    print(params)

    for i in range(len(datas[:,0,0])):
        data = datas[i,:,:]
        print(data)
        data = data[:,0:-1]
        # nettoyage data < 0
        data[data < 0] = 0

        bounds = ((0.1,0.99),(0.1,0.99),(0.01,0.99))
        params0 = [0.5,0.3,0.01]
        # 'L-BFGS-B'
        minimizer_kwargs = {
            "args": (data),
            "bounds" : bounds,
            "method": 'L-BFGS-B'
            }


        # res = basinhopping(
        #     log_likelihood,
        #     x0=params0,
        #     minimizer_kwargs=minimizer_kwargs,
        # )
        #res = dual_annealing(log_likelihood,bounds=bounds,args=(data))
        params.append(res.get("x"))
        print(res)
    params = np.array(params)
    np.save("params.txt",params)
    
   