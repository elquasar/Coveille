from pickle import BINSTRING
import numpy as np
import matplotlib.pyplot as plt
import scipy 


plt.rcParams["figure.figsize"] = (20,8)

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from models.SIRD import SIRD
 
if __name__ == '__main__': 
    
    dt = 0.001
    sird = SIRD()
    N = 1e3
    # Paramètres du modèle 
    sird.setParam({
    'N': N,   # Population totale
    'beta': 0.6, # Taux de transmission   0 < beta  < 1   
    'gamma':0.1, # Taux de guérison (1/j) 0 < gamma < 1
    'nu':0.02   # Taux de létalité       0 < nu    < 1
    })
    # Vecteur de conditions initiales (S0,I0,R0,D0)
    X0 = np.array([N-10,10,0,0])
    T = 60
    t = np.linspace(0,T,int(T/dt)) 
    ##################################
    ##################################
    #      Simulation du modèle      #
    sird.simulate(
        X0,   # Conditions initiales
        t,    # Vecteur de temps
        dt,   # Pas de temps
        10, # Nombre de simulations
        type_simulation="all"
        )
    sird.aplot(dtmc=True,alpha=0.4,det=True,conf=True)
    plt.show()

    

    # data = sird.sol_dtmc[0]
    # chunks = np.split(data,T)
    # beta = np.zeros(len(chunks))
    # gamma = np.zeros(len(chunks))
    # nu = np.zeros(len(chunks))
    # for index,chunk in enumerate(chunks) : 
    #     Nsi = 0
    #     Nir = 0
    #     Nid = 0

    #     Ni = chunk[-1][1]
    #     Ns = chunk[-1][0]
        
    #     if Ni != 0 and Ns != 0 :
    #         for i in range(len(chunk)-1):
    #             if chunk[i+1][1] > chunk[i][1] and chunk[i+1][0] < chunk[i][0]:
    #                 Nsi += chunk[i][0] - chunk[i+1][0]
    #             if chunk[i+1][1] < chunk[i][1] and chunk[i+1][2] > chunk[i][2]:
    #                 Nir += chunk[i+1][2] - chunk[i][2]
    #             if chunk[i+1][1] < chunk[i][1] and chunk[i+1][3] > chunk[i][3]:
    #                 Nid += chunk[i+1][3] - chunk[i][3]
        
    #         beta[index] = (Nsi/(Ni*Ns))
    #         gamma[index] = (Nir/(Ni))
    #         nu[index] = (Nid/(Ni))
    
    # beta = beta*sird.parameters['N']
    # beta = beta[beta >0]
    # gamma = gamma[gamma >0]
    # nu = nu[nu >0]
    # # nettoyage nan
    # beta = beta[~np.isnan(beta)]
    # gamma = gamma[~np.isnan(gamma)]
    # nu = nu[~np.isnan(nu)]


    # print("Moyenne beta : ", np.mean(beta), " || Median beta :", np.median(beta))
    # print("Moyenne gamma : ", np.mean(gamma), " || Median gamma : ", np.median(gamma))
    # print("Moyenne nu : ", np.mean(nu), " || Median nu : ", np.median(nu))

    # def mean_confidence_interval(data, confidence=0.95):
    #     a = 1.0 * np.array(data)
    #     n = len(a)
    #     m, se = np.mean(a), scipy.stats.sem(a)
    #     h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    #     return m, m-h, m+h

    # print("Intervalle de confiance sur beta : ", mean_confidence_interval(beta))
    # print("Intervalle de confiance sur gamma : ", mean_confidence_interval(gamma))
    # print("Intervalle de confiance sur nu : ", mean_confidence_interval(nu))

    # fig,(ax1,ax2,ax3) = plt.subplots(3)
    # ax1.hist(beta,bins=100,label='test')
    # ax2.hist(gamma,bins=100)
    # ax3.hist(nu,bins=100)
    # plt.legend()
    # plt.show()
