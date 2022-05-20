from models.factory import Factory
import numpy as np
import matplotlib.pyplot as plt
import sys
liste_modele = {
    1 : "SIRD",
    2 : "SIRDV"
}

def confirmer_y_n(phrase):
    confirmation=""
    while confirmation != "y" or confirmation != "n" : 
        confirmation = str(input("{} ? (y/n) > ".format(phrase)))
        if confirmation == "y":
            return True
        if confirmation == "n":
            return False
        if confirmation != 'y' or confirmation!= 'n':
            print("Veuillez entrer y ou n.")

def ask_param(parameters):
    while True : 
        print(parameters,": ")
        
        if parameters == 'N' :
            value = demander_nombre()
            if value is not None and value >= 1 :
                return value
            else : 
               print("Veuillez entrer un nombre supérieur à 0") 
        else : 
            value = demander_nombre()
            if value is not None and 0 <= value <= 1:
                return value
            else :
                print("Veuillez entrer un nombre entre 0 et 1")


def affiche_list_model():
    print("Veuillez choisir une modèle parmi cette liste : ")
    for keys, values in liste_modele.items():
        print(keys,"-",values)
    print("3 - Quitter le programme")

def affichage_bienvenue(): 
    print("###################################################")
    print("Module interactif de simulation d'épidemie Coveille")
    print("###################################################")

def valid_model(choix_model):
    if choix_model in liste_modele.keys():
        return True
    else:
        return False

def demander_nombre():
    try : 
        choix_model = float(input("> "))
        return choix_model
    except ValueError:
        print("Veuillez entrer un nombre")
        return demander_nombre()

def affichage_option_model():
    print("Veuillez choisir une option : ")
    print("1 - Charger un modèle existant")
    print("2 - Faire un nouveau modèle")
    print("3 - Revenir en arrière")

def main() : 


    def choix_mdl():
        affiche_list_model()
        while True:
            choix_model = demander_nombre()
            if valid_model(choix_model):
                mdl = f.give_model(liste_modele[choix_model])
                print(mdl)
                return mdl
            elif choix_model == 3:
                print("Fermeture du programme...")
                sys.exit()

    
    def choix_option():
        while True :
            choix = demander_nombre()       
            
            if choix == 3 :
                choix_mdl()
            if choix == 1:
                # fonction de chargement des modèles
                # --> jonction avec affichage
                break
            if choix == 2 :
                # fonction de ccréation modèle
                creation_model(mdl)
                break
    def creation_model(mdl):

        # paramètres du mdl
        print("Veuillez entrer la valeur du paramètre :")
        params = {}
        for parameters in mdl.parameters.keys():
            value = ask_param(parameters)
            params[parameters] = value
        mdl.setParam(params)
        creation_simu(mdl)

    def creation_simu(mdl):
        print("Veuillez entrer les paramètres de la simulation : ")
        
        T = 0 
        while T <= 0 :
            print("Entrer le temps de simulation en jours (T > 0) : ") 
            T = demander_nombre()

        dt = 0
        while dt > 1 or dt <= 0 :
            print("Entrer le pas de temps de la simulation (0 < dt < 1) : ")
            dt = demander_nombre()

        nsim = 0
        while nsim <= 0:
            print("Entrer le nombre de trajectoires à simulées : ") 
            nsim = int(demander_nombre())

        # Création de la grille de temps

        t = np.linspace(0,T, int(T/dt))
        print("Création du vecteur initial :")
        print("La somme des compartiment doit valoir {}".format(mdl.parameters['N']))
        # Vecteur X0 initial
        # Appelle a getcompartiment pour récupérer le nom des compartiments

        taille = len(mdl.name)
        X0 = np.zeros(taille)
        comp_list = []
        sum = 0
        while sum != mdl.parameters['N'] : 
            for i in range(taille):

                comp_name = mdl._getCompartimentName(i)[0]
                comp_list.append(mdl._getCompartimentName(i)[0])
                value = int(input("{} > ".format(comp_name)))
                X0[i] = value
            sum = np.sum(X0)

            if sum != mdl.parameters['N'] : 
                print("Il faut que la somme des compartiments soit égale à {}".format(model.parameters['N']))

        if confirmer_y_n("Confirmez"): 
            simulation(mdl,X0,t,dt,nsim)
        else : 
            creation_model(mdl)


    def simulation(mdl,X0,t,dt,nsim):
        mdl.simulate(X0, t, dt, nsim)
        if confirmer_y_n("Souhaitez-vous sauvegarder la simulation"): 
            try : 
                mdl.save_sim()
            except Exception as e :
                print(e)
            print("La données de simulation sont situées dans le répertoire simulation/")
            print("Vous pourrez charger cette simulation plus tard")
            print(" /!\ Le graphe de la solution a été sauvegardé /!\ ")
        if confirmer_y_n("Afficher la solution"):
            restart = True
            while restart : 
                print("Selectionner mode affichage : ")
                print("1. Trajectoires stochastiques")
                print("2. Solutions déterministes")
                print("4. Les deux")
                choix_mode = int(input(" > "))
                if choix_mode == 1 : 
                    for i in range(len(mdl.name)):
                        mdl.mplot(i, conf=True, det=False, stoch=True)
                if choix_mode == 2 : 
                    for i in range(len(mdl.name)):
                        mdl.mplot(i, conf=False, det=True, stoch=False) 
                if choix_mode == 3:
                    mdl.save_fig()

                if choix_mode == 4 : 
                    for i in range(len(mdl.name)):
                        mdl.mplot(i, conf=True, det=True, stoch=True)
                plt.show()
                if confirmer_y_n("Souhaitez-vous choisir un autre mode d'affichage"):
                    restart = True
                else : 
                    choix_mdl()
        else :
            choix_mdl()
    def interface_load_sim():
        pass





    affichage_bienvenue()
    f = Factory()
    # Choix du modèle

    mdl = choix_mdl()

    # Charger un modèle existant
    affichage_option_model()
    # choix des options pour le modèle
    choix_option()

if __name__== '__main__':
    main()