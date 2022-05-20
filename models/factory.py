from .SIRD import SIRD
from .SIRDV import SIRDV 

class Factory :
    def give_model(self,name):
        
        if name == "SIRD":
            print("Création d'un modèle SIRD...")
            return SIRD() 
        if name == "SIRDV":
            print("Création d'un modèle SIRDV...")
            return SIRDV()
        else :
            raise Exception("Le nom spécifié ne correspond a aucun modèle existant.. n'hésitez pas à l'implémenter..")