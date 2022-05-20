
#########################################
# Import des libraries nécessaires
from models.tools.functions import plot_template
from typing import ValuesView
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import scipy.linalg
import scipy.stats
import os
import warnings
import pandas as pd
#########################################

warnings.filterwarnings("ignore", category=RuntimeWarning) 


class model:
    """
    Classe abstraite pour un modèle compartimental quelconque, 
    ne doit pas être instanciée.
    """

    def __init__(self, param):
        """CONSTRUCTOR

        Args:
            param (dict): Python dictionnary with format {string param_name : float value }
        """
        self.parameters = {}
        for element in param:
            if element not in self.parameters.keys():
                self.parameters[element] = 0
        self.name = 'Not yet named model'

        self.fitted = None
        self._isSim = False

        self.sol_deter = None
        self.sol_stoch = None
        self.sol_dtmc = None
        self.sol_ctmc = None


        self.info_sim = {}

    def setMeasures(self,start,stop,strength,type='exponential'):
        if start > stop or start < 0 or stop < 0 : 
            raise Exception("Erreur sur les durées des mesures sanitaires")
        self.measures = {
            "start" : start,
            "stop" : stop,
            "strength" : strength,
            "type" : type
        }



    def setName(self, name):
        """Donne un nom au modèle, le nom du modèle doit être exactement expliciter tous les compartiments.
        Args:
            name (string): Nom du modèle
        """
        self.name = name

    def __repr__(self):
        """Redéfinition de la représentation du modèle

        Returns:
            string: imprime le modèle avec le format suivant : 
                    'Modèle {nom du modèle}
                    paramètres: 
                    {param1} : value1,
                    ...
                    '{paramN}' : valueN'
        """

        output = ""
        output += "Modèle '{}' \n Init parameters : \n".format(self.name)
        for keys, values in self.parameters.items():
            output += "{} : {} \n".format(keys, values)
        return output

    def setParam(self, param):
        """Assigner les paramètres du modèle

        Args:
            param (dict): Dictionnaire de la forme : 
                        {'nom_parametre1' : valeur_parametre1,
                        ...
                        'nom_parametreN' : valeur_parametreN}

        Raises:
            Exception: A redéfinir dans toutes les classes filles
        """
        for keys, values in param.items():
            if keys not in self.parameters.keys():
                raise Exception("Le nom de paramètres ne correspond pas avec les paramètres du modèle : {}".format(
                    self.parameters.keys()))
            else:
                self.parameters[keys] = values

    def _setEnvSimulation(self, X0, t, dt, nsim, parameters=None):
        """Met à jour les paramètres de la simulation et les sauvegarde, 
            vérifie les paramètres

        Args:
            X0 (np.ndarray): Vecteur initial
            t (np.ndarray): Grille de temps
            dt (float): Pas de temps
            nsim (int): Nombre de simulation    
            parameters (dict, optional): Paramètres à modifier. Defaults to None.

        Returns:
            model: Retourne l'objet model mis à jour
        """
        if nsim < 1 :
            raise Exception("Le nombre de simulation doit être supérieur à 1")

        self._dt = dt
        self._X0 = X0
        self._t = t
        self._nsim = nsim
        if parameters is not None:
            self.setParam(parameters)
        return self

    def _checkFitted(self):
        return self.fitted is not None
    def _updateSim(self):
        """Signale qu'une simulation a déjà été faite
        """
        self._isSim = True


    def simulate(self, X0, t, dt, nsim=1,type_simulation="all",parameters=None,show_details=True):
        """Simule le modèle pour un vecteur X0 donné, une grille de temps, ainsi qu'un pas de temps.

        Args:
            X0 (np.ndarray): Vecteur initial
            t (np.ndarray): Grille de temps
            dt (float): Pas de temps de la simulation
            nsim (int, optional): Nombre de simulation dans le cas stochastique. Defaults to 1.
            parameters (dict, optional): Dictionnaire de paramètres à passer 
            à la fonction setParam. Defaults to None.
            type_simulation (str) : Type de simulations parmi (deterministe,stochastique,dtmtc,ctmc). Defaults to 'all'.
        Returns:
            np.ndarray,np.ndarray : Matrice contenant les solutions X(t) pour tout t dans 
            le cas déterministe et une matrice contenant les solutions X(t) pour tout t 
            et pour chaque simulation
        """
        self._setEnvSimulation(X0, t, dt, nsim, parameters)

        # mis à jour des paramètres si nécessaire.
        if parameters is not None:
            self.setParam(parameters)
        
        if show_details:
            ######################### TIMER START #######################
            FMT = '%H:%M:%S'
            start = time.time()
            start = datetime.datetime.fromtimestamp(start).strftime(FMT)
            #############################################################
        
        if show_details : 
            self._simuleSUMUP(X0, t, dt, nsim)

        if type_simulation == "all" : 
            self.sol_stoch = self.stochastique(X0, t, dt, nsim)
            self.sol_deter = self.deterministe(X0, t, dt)
            self.sol_dtmc = self.dtmc(X0,t,dt,nsim)
        elif type_simulation == "stoch" or type_simulation == 'stochastique':
            self.sol_stoch = self.stochastique(X0, t, dt, nsim)
        elif type_simulation == "deter" or type_simulation == 'deterministe':
            self.sol_deter = self.deterministe(X0, t, dt)
        elif type_simulation == "dtmc" : 
            self.sol_dtmc = self.dtmc(X0,t,dt,nsim)
        if show_details:
            ######################### TIMER END #########################

            print("Started at : ", start)

            finish = time.time()
            finish = datetime.datetime.fromtimestamp(finish).strftime(FMT)
            print("Finished at : ", finish)
            tdelta = datetime.datetime.strptime(
                finish, FMT) - datetime.datetime.strptime(start, FMT)
            print("Elapsed time : ", tdelta)
            print("Utilisez la méthode mplot() pour afficher graphiquement les résultats")
            #############################################################

        # update _isSim
        self._updateSim()


        return self

    def varestimator(self):
        """Estime la variance pour des paramètres de la simulation

        Returns:
            np.ndarray: Moyenne des variances de chaques trajectoires
        """

        solution = self.sol_stoch
        nsim = self._nsim
        dim = len(self._X0)
        n = len(self._t)
        T = self._dt * n
        print("-------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------")
        print("\n")
        print("     ==========================================================      ")
        print("           Estimation de la variance,moyenne de X                    ")
        print("                Nombre de trajectoires:",
              nsim, "                    ")
        print("                Temps de simulation : ",
              T, "                         ")
        print("                 Pas de temps : ",
              self._dt, "                        ")
        print("     ==========================================================      ")
        var_k = np.zeros((nsim, dim, dim))
        for k in range(nsim):
            for i in range(n-1):
                diff = solution[k, i+1] - solution[k, i]
                vec = np.array(diff)[np.newaxis]
                var_k[k] += vec*vec.T
            var_k[k] = var_k[k] * (1/(self._dt*n))

        # calcule de la moyenne des variances obtenues
        var_tot = np.mean(var_k, axis=0)
        print("         ============= Variance moyenne ==============       ")
        print(var_tot)
        print("         =============================================       ")
        print("-------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------")
        self.var = var_tot
        return self.var

    def _mean_confidence(self, solution, composante=0, confidence=0.95):
        """Calcul un intervalle de confiance en chaque temps pour une composante donnée

        Args:
            solution (np.ndarray): Matrice de solution obtenue via la méthode simulate()
            composante (int, optional): Composante du vecteur solution. Defaults to 0.
            confidence (float, optional): Niveau de confiance. Defaults to 0.95.
        """

        def mean_confidence_interval(data, confidence):

            a = 1.0 * np.array(data)
            m = np.mean(a)
            ci = scipy.stats.t.interval(alpha=confidence, df=len(
                data)-1, loc=np.mean(data), scale=scipy.stats.sem(data))

            return m, ci[0], ci[1]


        ic = np.zeros((len(solution[0]), 3))

        for index, sol in enumerate(np.transpose(self.sol_stoch[:, :, composante])):
            ic[index] = mean_confidence_interval(sol, confidence)

        return ic

    def mplot(self, compartiment=0, stoch=False, det=False, conf=False,alpha=None,dtmc=False,label=True,prop=False):
        """Permet l'affichage des solution de la méthode simulate(). \n
        On peut afficher les trajectoires stochastiques et la courbe déterministe
        ainsi que les intervalles de confiance sur les trajectoires stochastiques.\n
        -> Utiliser plt.show() après l'utilisation de mplot() pour générer l'affichage <-
        
        Args:
            compartiment (int, optional): Composante du vecteur solution a afficher. Defaults to 0.
            stoch (bool, optional): Affichage des trajectoires stochastique. Defaults to True.
            det (bool, optional): Affichage de la courbe déterministe. Defaults to False.
            conf (bool, optional): Affichage de l'intervalle de confiance sur la moyenne empirique des trajectoires stochastiques. Defaults to False.

        Returns:
            model: Retourne l'instance courante de model
        """

        compartiment_name, color = self._getCompartimentName(compartiment)
        
        nsim = self._nsim
        if alpha is None : 
            if nsim > 500:
                alpha = 0.01
            else:
                alpha = 0.1
        t = self._t
        
        only_dtmc = stoch == False and det == False and dtmc == True 
        only_sde  = stoch == True  and det == False and dtmc == False
        only_det  = stoch == False and det == True  and dtmc == False 

        if only_dtmc:
            plt.title("Simulation DTMC d'un modèle {}".format(self.name))

        if only_sde:
            plt.title("Simulation SDE d'un modèle {}".format(self.name))
        if only_det:
            plt.title("Simulation déterministe d'un modèle {}".format(self.name))        
        plt.grid()
        plt.xlabel("Temps")
        plt.ylabel("Population")


        if det and self.sol_deter is not None:
            plt.plot(t, self.sol_deter[:, compartiment], alpha=1, color=color,
                     label='Solution déterministe de {}'.format(
                         compartiment_name)
                     )
            
        if conf and self.sol_stoch is not None:
            # calcul intervalle de confiance moyenne empirique

            intervalle = self._mean_confidence(self.sol_stoch, compartiment)    
            plt.plot(t, np.transpose(intervalle)[
                     0], '--', label='Moyenne empirique de {}'.format(compartiment_name), color=color)
            plt.fill_between(t, np.transpose(intervalle)[1], np.transpose(intervalle)[
                             2], color=color, label='Intervalle de confiance sur la moyenne de {}'.format(compartiment_name), alpha=0.4)
        if stoch and self.sol_stoch is not None:
            plt.plot(t, np.transpose(
                self.sol_stoch[:, :, compartiment]), alpha=alpha, color=color)
        if dtmc and self.sol_dtmc is not None : 
            for i in range(len(self.sol_dtmc[:,0,0])): 
                plt.plot(t,self.sol_dtmc[i,:,compartiment],alpha=alpha,color=color)
        if label == True:
            plt.legend()
        
        if prop and self.sol_stoch is not None : 
            # variance
            var = []
            mean = []
            for i in range(len(self.sol_stoch[0,:,compartiment])):
                var.append(np.var(self.sol_stoch[:,i,compartiment]))
                mean.append(np.mean(self.sol_stoch[:,i,compartiment]))
            var = np.array(var)
            var = np.sqrt(var)
            mean = np.array(mean)
            f = 2
            if not conf : 
                plt.plot(t,mean,color=color,ls='--',label="Moyenne empirique")
            plt.fill_between(t,f*var+mean,mean-f*var,color=color,alpha = 0.2,label="Intervalle de proportion à 95%",hatch = "///")
          

        return self

    def aplot(self,stoch=False, det=False, conf=False,alpha=None,dtmc=False,prop=False,label=True):
        for i in range(len(self.name)):
            self.mplot(compartiment=i,stoch=stoch,det=det,conf=conf,alpha=alpha,dtmc=dtmc,label=label,prop=prop)
        return self
    def save_fig(self,dpi=400, bbox_inches='tight', pad=0.1):
        """
        Save the current figure as a file to disk.
        
        Arguments
        ---------
        filename: string
            filename of image file to be saved
        folder: string
            folder in which to save the image file
        dpi: int
            resolution at which to save the image
        bbox_inches: string
            tell matplotlib to figure out the tight bbox of the figure
        pad: float
            inches to pad around the figure
        
        Returns
        -------
        None
        """
        # Rechercher si le dossier simulations existe :
        # on se dirige dans le répertoire simulations,

        plt.tight_layout()
        plt.rcParams["figure.figsize"] = (20,8)
        self.aplot(det=True)

        plt.savefig('simu_det.png', dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad)

        plt.clf()

        self.aplot(stoch=True,conf=True)

        plt.savefig('simu_stoch.png', dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad)
        
        plt.clf()
        

    # PRIVATE

    def _getCompartimentName(self):
        """Retourne le compartiment et son nom, doit être implémenté dans les classes filles

        Raises:
            NotImplementedError: Méthode virtuelle
        """

        raise NotImplementedError("Subclass must implement abstract method")

    def _simuleSUMUP(self, X0, t, dt, nsim):
        """Résume les paramètres de la simulation

        Args:
            X0 (np.ndarray): Vecteur initial
            dt (float): Pas de temps
            nsim (int): Nombre de simulation pour le cas stochastique
        """
        output =""
        print("=============================================================")
        output +="============================================================="
        output +="\n"
        print("Integration d'un modèle {} stochastique...".format(self.name))
        output += "Modèle {} stochastique".format(self.name)
        output += "\n"
        print(" Nombre de simulation : {}".format(nsim))
        output += "Nombre de simulation : {}".format(nsim)
        output += "\n"
        print(" Durée de la simulation T = {} jours".format(len(t)*dt))
        output += "Durée de la simulation T = {} jours".format(len(t)*dt)
        output += "\n"
        print(" Pas de temps dt = {}".format(dt))
        output += " Pas de temps dt = {}".format(dt)
        output += "\n"
        print(" Paramètres :")
        output +=" Paramètres :"
        for key, value in self.parameters.items():
            output += "   {}  :  {}   ".format(key,value)
            print('  ', key, ' : ', value)
            output += "\n"
        print(" Vecteur initial X0 = {}".format(X0))
        output +=  "Vecteur initial X0 = {}".format(X0)
        output += "\n"
        print("=============================================================")
        output += "============================================================="
        return output

    def forecast(self,data,debut,fin,dt,nsim,method='deterministe') :
        """Permet de projeter les courbes du modèle en utilisant les paramètres calculés à une date donnée

        Args:
            data (pandas.DataFrame): Données parsé pour être injecté dans le modèle ['jour','S','I',...], les compartiments
            doivent être dans le bon ordre
            debut (datetime.datetime): Date de début
            fin (datetime.DateTime): Date de fin de la projection
            dt (float): Pas de temps de la simulation
            nsim (int): Nombre de simulation
            method (str, optional): Méthode de fit pour les données. Defaults to 'deterministe'.

        Returns:
            solution: Matrice tri-dimensionnelle de type [simulation:temps:compartiment]
        """
        # fit des données avec la méthode
        self.fit(data,method)
        # recuperer les valeurs des compartiments au bon jour
        data = data[data.jour == debut]
        data = data.drop('jour',axis=1)
        # transformation en dictionnaire
        
        data = data.to_dict('records')[0]
        
        # nourir le vecteur X0 inital de la valeur des 
        # compartiments au jour debut
        X0 = []
        for key, value in data.items() : 
            X0.append(value)
        #make numpy array
        X0 = np.array(X0)*self.parameters['N']

        # recuperer le vecteur de paramètre calculer le 
        # bon jour

        params = self.fitted
        # retirer le R0 calculé
        params = params.drop('R0',axis=1)
        # selectionner les paramètres en fonction du debut
        params = params[params.jour == debut]
        params = params.drop('jour',axis=1)
        params = params.to_dict('records')[0]
        # ajouter le paramètre N
        params['N'] = self.parameters['N']

        # lancer la simulation

        # temps de simulation : 
        T = fin - debut
        T = int(T.days)
        # grille de temps
        t = np.linspace(0,T,int(T/dt))

        # pret a être simuler

        self.simulate(X0,t,dt,nsim,parameters=params)
        sto = self.sol_stoch
        det = self.sol_deter

        # convertir les temps de simulation en date : 
        new_dates = []
        for i in range(len(t)) : 
            new_dates.append(debut+datetime.timedelta(i))
        for k in range(len(sto[:,0,0])): 
            plt.plot(new_dates,sto[k,:,1],color='k',alpha=0.1)


        
        
    def fplot(self,data):
        plot_template()
        # plt.plot(data.jour,data.S,label='I',color='r')
        plt.plot(data.jour,data.I,label='I',color='r')
        plt.plot(data.jour,data.R,label='R',color='g')        
        plt.plot(data.jour,data.D,label='D',color='y')
        plt.xlabel("Temps en jour")
        plt.ylabel("Proportion de la population")
        plt.legend()

    # méthode virtuelle
    def stochastique(self, X0, t, dt, nsim):
        raise NotImplementedError("Subclass must implement abstract method")

    def dtmc(self,X0,t,dt,nsim):
        raise NotImplementedError("Subclass must implement abstract method")

    def deterministe(self, X0, t, dt):
        raise NotImplementedError("Subclass must implement abstract method")

    def fit(self,data,method='deterministe'):
        
        """Fit les données au modèle pour l'estimation de ses paramètres

        Args:
            data (pd.DataFrame): DataFrame avec comme nom de colonnes : 
            ["jour","S","I","R","D",..."]
            renseignant sur la valeur des compartiments S,I,R et D pour chaque temps t
            option (str, optional): Mode de fitting. Defaults to 'deterministe'.

        Returns:
            pd.DataFrame : DataFrame qui donne les paramètres du modèle estimé pour chaque temps t
            de la forme : 
            ["jour","beta","gamma","nu","R0"]
        """
        # recoit des données sous la forme d'un dataframe avec en colonne les valeurs des compartiments
        # a chaque jour
        print("============================================================")
        print("Fitting data from : {} to {}".format(
            str(data[:1].jour.item()), str(data[-1:].jour.item())))

        if method == 'deterministe':
            self._fit_deterministe(data)
        return self.fitted

    def _fit_deterministe(self,data):
        return NotImplementedError

    def save_sim(self):

        today = datetime.date.today()
        
        if not os.path.exists("simulations"):
            os.makedirs("simulations")
        if not os.path.exists("simulations/sim{}".format(today)):
            os.makedirs("simulations/sim{}".format(today))
        
        # sauvegarder la simulation

        # créer un fichier qui porte le nom de l'heure/minute/seconde
        instant = datetime.datetime.now()
        
        if not os.path.exists("simulations/sim{}/sim{}-{}-{}".format(today,instant.hour,instant.minute,instant.second)):
            os.makedirs("simulations/sim{}/sim{}-{}-{}".format(today,instant.hour,instant.minute,instant.second))
        if self.sol_deter is not None : 

            np.save(r"simulations/sim{}/sim{}-{}-{}/sim_deterministe".format(today,instant.hour,instant.minute,instant.second), self.sol_deter)
        if self.sol_stoch is not None :
            np.save(r"simulations/sim{}/sim{}-{}-{}/sim_sde".format(today,instant.hour,instant.minute,instant.second), self.sol_stoch)
        if self.sol_dtmc is not None :
            np.save(r"simulations/sim{}/sim{}-{}-{}/sim_dtmc".format(today,instant.hour,instant.minute,instant.second), self.sol_dtmc)
        with open(r"simulations/sim{}/sim{}-{}-{}/info.txt".format(today,instant.hour,instant.minute,instant.second),mode="w") as info:
            output = ""
            output += "nsim={}".format(self._nsim) 
            output += "\n"            
            output += "T={}".format(int(len(self._t)*self._dt)) 
            output += "\n"
            output += "dt={}".format(self._dt) 
            output += "\n"
            output += "X0={}".format(self._X0)
            # a partir de là , ajouter les paramètres
            output += "\n"
            for keys,values in self.parameters.items():
                output += "{}={}".format(keys,values) 
                output += "\n"
        
            info.write(output)
        # déplacement dans le fichier courant : 
        os.chdir(r"simulations/sim{}/sim{}-{}-{}".format(today,instant.hour,instant.minute,instant.second))
        if not os.path.exists("Figures"):
            os.makedirs("Figures")
        os.chdir("Figures")
        self.save_fig()
        

    def load_model_sim(self,path):
        
        
        params = {}
        
        # ouverture d'un dossier de simulation 
        os.chdir(path)
        print("Vous êtes au {}".format(os.getcwd()))

        # ouverture résumé simulation
        with open(r"info.txt",mode="r") as info :
            print("Lecture du fichier info.txt...")

            lines = info.readlines()
            lines = [line.replace('\n', '') for line in lines]

            for line in lines:
                print(line)

            nsim = int(lines[0].split('=')[-1])
            T = float(lines[1].split('=')[-1])
            dt = float(lines[2].split('=')[-1])
            X0 = lines[3].split('=')[-1]
            X0 = X0[1:-1]
            X0 = X0.split()
            X0 = [float(v) for v in X0]
            t = np.linspace(0,T,int(T/dt))

            self._setEnvSimulation( X0, t, dt, nsim, parameters=None)
            for line in lines[4:]:
                p = line.split("=")
                params[str(p[0])] = float(p[1])
                self.setParam(params)
            print("Les paramètres du modèle pour cette simulation ont été restauré")

            print("Chargement des simulations...")
            # Verifier que le fichier existe bel et bien.
            try : 
                self.sol_stoch = np.load("sim_sde.npy")
            except FileNotFoundError:
                print("--> Le modèle SDE n'a pas encore été simulé")

            try : 
                self.sol_dtmc = np.load("sim_dtmc.npy")
            except FileNotFoundError:
                print("-->Le modèle DTMC n'a pas encore été simulé")

            try : 
                self.sol_deter = np.load("sim_deterministe.npy")
            except FileNotFoundError:
                print("-->Le modèle déterministe n'a pas encore été simulé")
                
            print("> Les simulations trouvées du modèle ont été restauré")


    def save_model(self):
        if not os.path.exists("simulations"):
            os.makedirs("simulations") 
