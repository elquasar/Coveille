import numpy as np
import matplotlib.pyplot as plt
import datetime

from models.SIRD import SIRD
from models.SIRDV import SIRDV

from models.tools.constant import DATE_FRANCE,N
from models.tools.functions import data_parser_SPF, give_compartiment
plt.rcParams["figure.figsize"] = (18,8)



if __name__ == '__main__':

    #################################
    #  Paramètres de la simulation  #

    T = 40                         # Temps de simulation en jour
    dt = 0.1                        # Pas de temps de la simulation
    nsim = 100                     # Nombre de trajectoires simulées pour la simulation stochastique
    # /!\ NE PAS MODIFIER /!\
    n = int(T/dt)                   # Nombre de points simulés
    t = np.linspace(0,T,int(T/dt))  # Vecteur de temps
    #################################
    
    N = 67e6

    #################################
    ##   Instanciation d'un modèle ##
            # MODÈLE SIRD #
    sird = SIRD()
    # Paramètres du modèle 
    # sird.setParam({
    # 'N': 67e6,   # Population totale
    # 'beta':0.8, # Taux de transmission   0 < beta  < 1   
    # 'gamma':0.3, # Taux de guérison (1/j) 0 < gamma < 1
    # 'nu':0.02   # Taux de létalité       0 < nu    < 1
    # })
    # Vecteur de conditions initiales (S0,I0,R0,D0)
    # X0 = np.array([N-10,10,0,0]) 
    ##################################
    ##################################
    #      Simulation du modèle      #
    # sird.simulate(
    #     X0,   # Conditions initiales
    #     t,    # Vecteur de temps
    #     dt,   # Pas de temps
    #     nsim, # Nombre de simulations
    #     type_simulation="stochastique"
    #     )
    ##################################
    data = give_compartiment('SIRD')
    data = data_parser_SPF()
#     # print(data)
#     # sird.fplot(data)
#     # start = data.iloc[-1].jour
#     # stop = start + datetime.timedelta(days=10)
#     #sird.forecast(data,data.iloc[-1].jour,stop,dt=1,nsim = 1000,method='deterministe')
#     # sird.mplot(compartiment=1,stoch=True,prop=True)
#     sird.setParam({'N' : N})
#     sird.fit(data)
#     params = sird.fitted[['beta','gamma','nu']].to_numpy()
#     print(data)

#     data = data[['S','I','R','D']]
#     data = np.array(data)
    
#     data = data*N
#     np.save("sird",data)
#     print(data)
#     # for i in range(350,len(params)-71):
        
#     #     if params[i][2] == 0:
#     #         params[i][2] = 1e-3
#     #     sird.setParam({
#     #         'beta': params[i][0],
#     #         'gamma': params[i][1],
#     #         'nu': params[i][2]
#     #     })
#     #     print(sird.parameters)
#     #     X0 = data.iloc[i][['S','I','R','D']]
#     #     X0 = X0.to_numpy()
#     #     X0 = X0*N
#     #     dt = 0.01
#     #     n = int(1/dt)
#     #     t = np.linspace(i,i+1,n)


#     #     sird.simulate(X0,t,dt,nsim=100,type_simulation="stoch",show_details=False)
#     #     sird.mplot(compartiment=1,stoch=True,prop=True,label=False)

#     plt.show()
#     # print(params)



#     ##################################
#     # Affichage des courbes simulées #

#     # Affichage compartiment S (composante 0)
#     # sird.mplot(
#     # compartiment = 0,   # Choix du compartiment
#     # stoch=True,        # Afficher les trajectoires stochastiques
#     # det=False,           # Afficher la solution déterministes
#     # conf=False,          # Afficher les intervalles de confiance sur la moyenne          # Option d'opacité des trajectoires stochastiques
#     # dtmc=False)         # Afficher la simulation Discret Time Markov Chain
#     # # Affichage compartiment I (composante 1)
#     # sird.mplot(1,stoch=True, det=False, conf=False, dtmc=False)

#     # # Affichage compartiment R (composante 2)
#     # sird.mplot(2,stoch=True, det=False, conf=False,dtmc=False,prop=True)
    
#     # # Affichage compartiment D (composante 3)
#     # sird.mplot(0,stoch=True, det=False, conf=False, alpha=0.2, dtmc=False)
#     # sird.aplot(stoch=True, det=False, conf=True, prop=True)
#     # # #/i\ Commenter pour retirer une courbe /i\
#     # # #Utiliser plt.show() pour afficher la figure
#     plt.show()
    
#     # c = 0 
#     # data = sird.sol_stoch
#     # var = []
#     # for i in range(len(data[0,:,c])):
#     #     var.append(np.var(data[:,i,c]))
#     # var = np.array(var)
#     # sig = np.sqrt(var)
#     # f = 2
    
#     # mean = []
#     # for i in range(len(data[0,:,c])):
#     #     mean.append(np.mean(data[:,i,c]))
#     # mean = np.array(mean)

#     # tps = [i for i in range(len(data[0,:,c]))]
    
    
    
#     # plt.plot(tps,mean,color='k',ls='--',label='Moyenne $\mu$')
#     # plt.fill_between(tps,f*sig+mean,mean-f*sig,color='k',alpha = 0.5,label="$\mu$ $\pm$ 2 $\sigma$ ")
#     # plt.legend()
#     # plt.show()
    
    
    
#     ################################### 
#     # sird.save_sim()
#     ###################################
#     #   Sauvegarde de la simulation   # 
#     # sird.save_sim()