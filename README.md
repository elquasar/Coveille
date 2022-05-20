# Développement d'outil d'aide à la simulation numérique
 -- Modèle type SIR déterministes et stochastiques -- 

!!! En construction !!!
# Présentation du projet 

Ce projet offre la possibilité de simuler quelques modèles compartimentaux. 
Les modèles disponibles sont les suivants : 
* SIR 
* SIRD
* SIRDV : vaccination

La formalisation des modèles est disponible dans le document infos/Formalisation_modèle_Compartimentaux.pdf, 

Outre la simulation, le projet dispose d'un parser, qui permet de traiter et récupérer les données de SPF en rapport avec l'épidémie de COVID-19.
Les données accessibles par le parser sont dans le dossier infos/parser_data.

Le code de ce parser est disponible dans functions.py

Grâce à ces données on peut faire l'estimation des différents paramètres des modèles. 

Il est possible de créer son propre modèle, en suivant le patron de classe dans le dossier template/model_compartiment.py

# Installation via Git

1. Avoir une installation de Git sur son système.
2. Créer un dossier et initialiser un repo git via git init dans un terminal
3. Cloner le repository via git clone "https://gitlab.utc.fr/lepratqu/covid-lmac.git"
4. Dans le fichier script.py, on peut alors importer et instancier nos modèles (voir template/exemple_script.py)

# Utiliser le module sans programmer

Il est possible d'executer le script interactive.py, qui permet de faire la simulation des modèles via une interface dans un terminal. 
Cependant toutes les méthodes ne sont pas implémentées, pour un usage plus poussé il faut s'inspirer du script "sim_SIRD.py". 
