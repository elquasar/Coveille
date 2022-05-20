"""
Ce fichier contient les constantes pour les simulations, par exemple les périodes de confinement pour différent pays
"""
import datetime
import pandas as pd
### Constante de temps (confinement, couvre-feu)
import datetime 
DATE_FRANCE = {
    'Confinement1' : [datetime.datetime(2020,3,17),datetime.datetime(2020,5,11)],
    'Confinement2' : [datetime.datetime(2020,10,29),datetime.datetime(2020,12,15)],
    'Confinement3' : [datetime.datetime(2021,3,31),datetime.datetime(2021,5,3)],
    'Feu18h' : [datetime.datetime(2021,1,14),datetime.datetime(2021,3,20)],
    'Feu19h' : [datetime.datetime(2021,3,20),datetime.datetime(2021,5,19)]
}

DATE_FRANCE = pd.DataFrame(DATE_FRANCE)
DATE_FRANCE.index = ["Debut","Fin"]

### Population frnaçaise

N = 67e6