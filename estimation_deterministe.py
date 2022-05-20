import numpy as np
import matplotlib.pyplot as plt



plt.rcParams["figure.figsize"] = (20,8)

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from models.SIRD import SIRD
 
if __name__ == '__main__': 
    
    dt = 0.01
    sird = SIRD()
    N = 1e3
    # Paramètres du modèle 
    sird.setParam({
    'N': N,   # Population totale
    'beta':0.8, # Taux de transmission   0 < beta  < 1   
    'gamma':0.3, # Taux de guérison (1/j) 0 < gamma < 1
    'nu':0.02   # Taux de létalité       0 < nu    < 1
    })
    # Vecteur de conditions initiales (S0,I0,R0,D0)
    X0 = np.array([N-10,10,0,0])
    T = 40
    t = np.linspace(0,T,int(T/dt)) 
    ##################################
    ##################################
    #      Simulation du modèle      #
    sird.simulate(
        X0,   # Conditions initiales
        t,    # Vecteur de temps
        dt,   # Pas de temps
        1, # Nombre de simulations
        type_simulation="deterministe"
        )
 
    
    # Estimation par optimisation d^'une fonction cible
    data = sird.sol_deter
    def mean_squared_error(params,data):  
        sird = SIRD()
        sird.setParam({
        'N': N,   
        'beta':params[0],
        'gamma':params[1],
        'nu': params[2]
        })
        # Vecteur de conditions initiales (S0,I0,R0,D0)
        X0 = data[0,:] 

        sird.simulate(X0,t,dt,type_simulation='deterministe',show_details=False)
        target = np.mean(norm(data-sird.sol_deter,axis=1))
        return target
    params0 = [0.5,0.5,0.5]
    bounds = ((1e-5,0.99),(1e-5,0.99),(1e-5,0.99))

    res = minimize(
    mean_squared_error,
        params0,
        args=(data),
        method='L-BFGS-B',
        bounds=bounds
    )
    print("Estimation minisation fonction cible : ", res.x)

    # estimation par méthode inverse
    beta = []
    gamma = []
    nu = []
    
    for i in range(len(data)-1):
        beta.append(
            N*(data[i,0] - data[i+1,0])/(dt*data[i,0]*data[i,1])
            )
        gamma.append(
                (data[i+1,2]-data[i,2])/(dt*data[i,1])
            )
        nu.append(
                (data[i+1,3]-data[i,3])/(dt*data[i,1])
            )
    beta = np.array(beta)
    gamma = np.array(gamma)
    nu = np.array(nu)


params = np.array([sird.parameters['beta'],sird.parameters['gamma'], sird.parameters['nu']])
mean_estimation = np.array([beta.mean(),gamma.mean(),nu.mean()])
print("Estimation méthode inverse : ", mean_estimation)