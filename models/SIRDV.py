


"""
Ce fichier implémente le modèle SIRD.

Description du modèle : 

S: Susceptible
I : Infected
R : Recovered
D : Death
V : vaccined

Paramètres du modèle : 

beta  : taux de transmission du virus
gamma : taux de guérison du virus
nu    : taux de létalité du virus
phi   : taux de vaccination
sigma : taux d'efficacité des vaccins

Graphe du modèle :

 -----       -----        -----
|  S  | --> |  I  | ---> |  R  |
 -----       -----        -----
   |         ^   |
   |         |   |
   |         |   |          -----
   |         |   | ----->  |  D  |
   |       -----            -----
   | ---> |  V  |
           -----
Utilisation du modèle : 

Ce modèle est viable à court terme après les premières vaccinations,
au dela d'une certaine période, l'immunisation grâce au vaccins peut 
perdre en efficacité

"""
#########################################
# Import des library officielles

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg
import scipy.stats

#########################################
#########################################

# Import des fichiers nécessaire du projet
from .model import model

from .tools.functions import plot_template

#########################################


class SIRDV(model):

    def __init__(self):
        super().__init__(param=['N', 'beta', 'gamma', 'nu', 'phi', 'sigma'])
        self.setName("SIRDV")

    def setSigma(self,sigma):
        self.parameters['sigma'] = sigma
    def __repr__(self):
        output= """

Description du modèle : 

S: Susceptible
I : Infected
R : Recovered
D : Death
V : vaccined

Paramètres du modèle : 

beta  : taux de transmission du virus
gamma : taux de guérison du virus
nu    : taux de létalité du virus
phi   : taux de vaccination
sigma : taux d'efficacité des vaccins

Graphe du modèle :

 -----       -----        -----
|  S  | --> |  I  | ---> |  R  |
 -----       -----        -----
   |         ^   |
   |         |   |
   |         |   |          -----
   |         |   | ----->  |  D  |
   |       -----            -----
   | ---> |  V  |
           -----
Utilisation du modèle : 

Ce modèle est viable à court terme après les premières vaccinations,
au dela d'une certaine période, l'immunisation grâce au vaccins peut 
perdre en efficacité

"""
        return output


    def _getCompartimentName(self, compartiment):
        """Retourne le nom du compartiment et sa couleur

        Args:
            compartiment (int): Numéro de la composante du vecteur solution

        Raises:
            Exception: On doit spécifier une composante entre 0 et 3

        Returns:
            string,string: 'Nom du compartiment','couleur du compartiment'
        """

        if compartiment == 0:
            return 'S', 'b'
        elif compartiment == 1:
            return 'I', 'r'
        elif compartiment == 2:
            return 'R', 'g'
        elif compartiment == 3:
            return 'D', 'y'
        elif compartiment == 4:
            return 'V', 'm'
        else:
            raise Exception(
                "La composante de la solution ne correspond a aucun compartiment")

    def _fit_deterministe(self, data):
        """Fitting des données avec la méthode déterministe

        Args:
            data (pd.DataFrame): DataFrame avec comme nom de colonnes : 
            ["jour","S","I","R","D","V"]
            renseignant sur la valeur des compartiments S,I,R,D et V pour chaque temps t

        Returns:
            pd.DataFrame : DataFrame qui donne les paramètres du modèle estimé pour chaque temps t
            de la forme : 
            ["jour","beta","gamma","nu","R0"]
        """
        sigma = self.parameters['sigma']
        self.fitted = []
        jours = data['jour']
        SIRDV = data

        for n, jour in enumerate(jours[:-1]):

            # système a l'état n
            SIRDV_n = SIRDV[SIRDV['jour'] == jour]

            # système à l'état n+1

            SIRDV_n_1 = SIRDV[SIRDV['jour'] == jours[n+1]]

            # vecteur des différences

            B = np.array([
                float(SIRDV_n_1['S']) - float(SIRDV_n['S']),
                float(SIRDV_n_1['R']) - float(SIRDV_n['R']),
                float(SIRDV_n_1['D']) - float(SIRDV_n['D']),
                float(SIRDV_n_1['V']) - float(SIRDV_n['V'])
            ])

            # matrice du sytème linéaire

            A = np.array([
                [-float(SIRDV_n_1['S'])*float(SIRDV_n_1['I']), 0, 0,-float(SIRDV_n_1['S'])],
                [0, float(SIRDV_n_1['I']), 0, 0],
                [0, 0, float(SIRDV_n_1['I']), 0],
                [self.parameters['sigma'] *
                    float(SIRDV_n['V'])*float(SIRDV_n_1['I']), 0, 0, float(SIRDV_n_1['S'])]
            ])

            # vecteur contenant tauj+1 et gammaj+1,nuj+1
            # B = AX <=> X = A^-1 * B
            X = np.matmul(np.linalg.inv(A), B)

            self.fitted.append({
                'jour': jours[n+1],
                'beta': X[0],
                'gamma': X[1],
                'nu': X[2],
                'phi': X[3],
                'R0':  X[0]* (SIRDV_n['S'] + sigma*SIRDV_n['V'])/ (X[1] + X[2])
            })
        self.fitted = pd.DataFrame(self.fitted)

        return self.fitted

    def _fdrift(self, X):
        """Fonction de drift du modèle SIRDV

        Args:
            X (np.ndarray): Vecteur de solution à l'instant t

        Returns:
            np.ndarray: Fonction de drift appliqué à X(t)
        """
        N     = self.parameters['N']
        beta  = self.parameters['beta']
        gamma = self.parameters['gamma']
        nu    = self.parameters['nu']
        phi   = self.parameters['phi']
        sigma = self.parameters['sigma']
        
        p1 = beta*X[0]*X[1] / N
        p2 = sigma*beta*X[4]*X[1] / N
        p3 = gamma*X[1] 
        p4 = nu*X[1]
        p5 = phi*X[0]
        return np.array([
            -p1 - p5,
            p1 + p2 -p3 -p4,
            p3,
            p4
        ])
    
    def _fdiffusion(self, X):
        """Matrice de diffusion du modèle SIRDV

        Args:
            X (np.ndarray): Vecteur de solution à l'instant t

        Returns:
            np.ndarray: Matrice de diffusion appliquée à X(t)
        """

        N     = self.parameters['N']
        beta  = self.parameters['beta']
        gamma = self.parameters['gamma']
        nu    = self.parameters['nu']
        phi   = self.parameters['phi']
        sigma = self.parameters['sigma']
        
        p1 = beta*X[0]*X[1] / N
        p2 = sigma*beta*X[4]*X[1] / N
        p3 = gamma*X[1] 
        p4 = nu*X[1]
        p5 = phi*X[0]

        if X[1] == 0 :

            a = np.array([
                [p1 + p5, -p1,0,0],
                [-p1,p1 + p2 + p3 + p4,-p3, -p4],
                [0,-p3,p3,0],
                [0,-p4,0,p4]
            ])
            a = scipy.linalg.sqrtm(a)
        else :
            a = np.array([
                [np.sqrt(p5),0,0,0],
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,0]
            ])


        return a


    def stochastique(self, X0, t, dt, nsim):
        """Résolution du système d'équadiff stochastiques

        Args:
            X0 (np.ndarray): Vecteeur initial
            t (np.ndarray): Grille de temps
            dt (float): Pas de temps
            nsim (int): Nombre de simulation

        Returns:
            np.ndarray: Matrice des solutions du sytèmes pour chaque instant t et pour chaque simulation
        """
        dim = len(X0)
        n = len(t)
        solution = np.zeros((nsim, n, dim))
        def eta(size): return np.random.normal(0, 1, size)[np.newaxis]
        for _ in range(nsim):

            # Matrice Nbre de point*3,contenant les solutions de X a l instant t
            M = np.zeros((n, dim))
            # Vecteur solution pour chaque simulation
            # t = 0 initialisation du vecteur
            M[0] = X0
            N_pop = self.parameters['N']
            # calcul de la solution
            for i in range(n-1):
                # Euler-Maruyama
                M[i+1][0:-1] = M[i][0:-1] + self._fdrift(M[i])*dt + np.sqrt(dt)*np.dot(
                    self._fdiffusion(M[i]), eta(dim-1).T).reshape(1, dim-1)[0]
                M[i+1][-1] = N_pop - np.sum(M[i+1][0:-1])
            solution[_] = M
        return solution


    def deterministe(self, X0, t, dt):
        """Résolution du système d'équadiff déterministe

        Args:
            X0 (np.ndarray): Vecteur initial
            t (np.ndarray): Grille de temps
            dt (float): Pas de temps

        Returns:
            np.ndarray: Matrice de solution à chaque instant t
        """
        N = self.parameters['N']
        # cumsum = np.sum(X0)
        # if cumsum != self.parameters[0]:
        #     raise Exception("N différent de la somme cumulée de X0")
        dim = len(X0)
        # nombre de point = longueur de l'intervalle / dt
        n = len(t)
        solution = np.zeros((n, dim))
        solution[0] = X0
        for i in range(0, n-1):
            solution[i+1][0:-1] = solution[i][0:-1] + \
                dt*self._fdrift(solution[i])
            solution[i+1][-1] = N - np.sum(solution[i+1][0:-1])
        return solution
    
    def dtmc(self, X0, t, dt, nsim):
        return 0
    def pplot(self,*args):
        """Graphique des paramètres fittés du modèle

        Raises:
            Exception: Il faut veiller à fitter le modèle via la méthode fit() avant.
        """
        if not self._checkFitted():
            raise Exception("Veuillez d'abord utiliser la méthode fit() pour pouvoir dessiner les paramètres")

        print("Utilisez plt.show() pour afficher le graphique")
        fig,ax = plot_template()
        ax.set_xlabel("Temps en jours")
        ax.set_ylabel("Valeur du paramètre estimée")
        # paramètre choisi
        if 'beta' in args :
            ax.plot(self.fitted.jour,self.fitted.beta,label='beta', color='r')
        if 'gamma' in args : 
            ax.plot(self.fitted.jour,self.fitted.gamma,label='gamma', color='g')
        if 'nu' in args : 
            ax.plot(self.fitted.jour,self.fitted.nu,label='nu', color='y')
        if 'phi' in args : 
            ax.plot(self.fitted.jour,self.fitted.phi,label='phi',color='m')
        if 'R0' in args :
            ax.plot(self.fitted.jour,self.fitted.R0,label='R0 effectif ($R_{t}$)',color='k')
            ax.axhline(1,label='$R0 = 1$')
        ax.legend()


