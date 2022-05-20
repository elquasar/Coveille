"""
Ce fichier implémente le modèle SIRD.

Description du modèle : 

S: Susceptible
I : Infected
R : Recovered
D : Death

Paramètres du modèle : 

beta  : taux de transmission du virus
gamma : taux de guérison du virus
vu    : taux de létalité du virus

Graphe du modèle :

 -----       -----        -----
|  S  | --> |  I  | ---> |  R  |
 -----       -----        -----
               |
               |          -----
               | ----->  |  D  |
                          -----
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


class SIRD(model):

    def __init__(self):
        super().__init__(param=['N', 'beta', 'gamma', 'nu'])
        self.setName("SIRD")


    def __repr__(self):
        output ="""
Description du modèle : 

S: Susceptible
I : Infected
R : Recovered
D : Death

Paramètres du modèle : 

beta  : taux de transmission du virus
gamma : taux de guérison du virus
vu    : taux de létalité du virus

Graphe du modèle :

-----       -----        -----
|  S  | --> |  I  | ---> |  R  |
-----       -----        -----
            |
            |          -----
            | ----->  |  D  |
                        -----
        """
        return output
# ====================================================================================================
# =================================METHODE PRIVEE=====================================================

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
        else:
            raise Exception(
                "La composante de la solution ne correspond a aucun compartiment")

    def _fit_deterministe(self, data):
        """Fitting des données avec la méthode déterministe

        Args:
            data (pd.DataFrame): DataFrame avec comme nom de colonnes : 
            ["jour","S","I","R","D"]
            renseignant sur la valeur des compartiments S,I,R et D pour chaque temps t

        Returns:
            pd.DataFrame : DataFrame qui donne les paramètres du modèle estimé pour chaque temps t
            de la forme : 
            ["jour","beta","gamma","nu","R0"]
        """

        self.fitted = []
        jours = data['jour']
        SIRD = data

        for n, jour in enumerate(jours[:-1]):

            # système a l'état n
            SIRD_n = SIRD[SIRD['jour'] == jour]

            # système à l'état n+1

            SIRD_n_1 = SIRD[SIRD['jour'] == jours[n+1]]

            # vecteur des différences

            B = np.array([
                float(SIRD_n_1['S']) - float(SIRD_n['S']),
                float(SIRD_n_1['R']) - float(SIRD_n['R']),
                float(SIRD_n_1['D']) - float(SIRD_n['D'])
            ])

            # matrice du sytème linéaire

            A = np.array([
                [-float(SIRD_n_1['S'])*float(SIRD_n_1['I']), 0, 0],
                [0, float(SIRD_n_1['I']), 0],
                [0, 0, float(SIRD_n_1['I'])]
            ])

            # vecteur contenant tauj+1 et gammaj+1,nuj+1
            # B = AX <=> X = A^-1 * B
            X = np.matmul(np.linalg.inv(A), B)

            self.fitted.append({
                'jour': jours[n+1],
                'beta': X[0],
                'gamma': X[1],
                'nu': X[2],
                'R0': X[0] / (X[1] + X[2])
            })
        self.fitted = pd.DataFrame(self.fitted)

        return self.fitted

    def _fdrift(self, X):
        """Fonction de drift du modèle SIRD

        Args:
            X (np.ndarray): Vecteur de solution à l'instant t

        Returns:
            np.ndarray: Fonction de drift appliqué à X(t)
        """
        # Si il n'y a plus d'infecté on retourne 0
        
        if X[1] <= 0:
            return np.zeros(len(X))
        N = self.parameters['N']
        beta = self.parameters['beta']
        gamma = self.parameters['gamma']
        nu = self.parameters['nu']

        return np.array([
            -beta*X[0]*X[1]/N,
            (beta*X[0]*X[1]/N)-gamma*X[1] - nu*X[1],
            gamma*X[1]
        ])

    def _fdiffusion(self, X):
        """Matrice de diffusion du modèle SIRD

        Args:
            X (np.ndarray): Vecteur de solution à l'instant t

        Returns:
            np.ndarray: Matrice de diffusion appliquée à X(t)
        """
        # Si il n'y a plus d'infecté on retourne 0

        if X[1] <= 0:
            return np.zeros((len(X),len(X)))
        N = self.parameters['N']
        beta = self.parameters['beta']
        gamma = self.parameters['gamma']
        nu = self.parameters['nu']
        a = np.array([
            [beta*X[0]*X[1]/N, -beta*X[0]*X[1]/N, 0],
            [-beta*X[0]*X[1]/N, (beta*X[0]*X[1]/N) + gamma *
             X[1] + nu*X[1], -gamma*X[1]],
            [0, -gamma*X[1], gamma*X[1]]
        ])
        return scipy.linalg.cholesky(a,lower=True)


# ====================================================================================================
# ====================================================================================================

# ====================================OVERRIDE FONCTION VIRTUELLES====================================
# ====================================================================================================


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
                M[i+1][0:-1] = M[i][0:-1] + self._fdrift(M[i][0:-1])*dt + np.sqrt(dt)*np.dot(
                    self._fdiffusion(M[i][0:-1]), eta(dim-1).T).reshape(1, dim-1)[0]
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
        N_pop = self.parameters['N']
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
                dt*self._fdrift(solution[i][0:-1])
            solution[i+1][-1] = N_pop - np.sum(solution[i+1][0:-1])
        return solution
    
    def dtmc(self, X0, t, dt, nsim):
        
        # Paramètres de la simulation
        N     = self.parameters['N'] # population totale 
        n     = len(t)               # nombre d'itérations
        # Paramètres du modèle
        beta  = self.parameters['beta']   *dt     # taux de transmission 
        gamma = self.parameters['gamma']  *dt     # taux de guérison (10 jours de guérison)
        nu    = self.parameters['nu']     *dt     # taux de mortalité 
        # Conditions initiales  

        M = np.zeros((n,4))               # matrice solution du système
                                          # à chaque itérations
        M[0,:] = X0                       # Initialisation de la matrice solution
        Z = np.zeros((nsim,n,4))          # matrice solution du système pour 
                                          # chaque trajectoires simulées
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
        return Z
            
        
    
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
        if 'R0' in args :
            ax.plot(self.fitted.jour,self.fitted.R0,label='R0 effectif ($R_{t}$)',color='k')
            ax.axhline(1,label='$R0 = 1$')
        ax.legend()



        
