# Development of numerical simulation tools
 -- Deterministic and stochastic SIR model -- 

# Presentation of the project 

This project offers the possibility to simulate some compartmental models. 
The available models are the following: 

* SIR 
* SIRD
* SIRDV: vaccination

The formalization of the models is available in the document infos/On_disease_spreading_models.pdf (written in french, but equations are universal :) )

In addition to the simulation, the project has a parser, which allows to process and retrieve SPF data related to the COVID-19 epidemic.
The data accessible by the parser are in the folder infos/parser_data.
Note that these data are public moreover the parser might be outdated if there are potentials updates on the way Sante Public France publish their data. 

The code of this parser is available in functions.py. 

Thanks to this data we can estimate the different parameters of the models. 


# Installation via Git

1. Have a Git installation on your system.
2. Create a folder and initialize a git repo via git init in a terminal
3. Clone the repository 
4. In the script.py file, we can then import and instantiate our models and play with them. 

# Using the module without programming

It is possible to execute the interactive.py script, which allows the simulation of models via an interface in a terminal. 
However, all the methods are not implemented, for a more advanced use, you must be inspired by the "sim_SIRD.py" script. 

The source code of this project contains the basis for the understanding and simulation of stochastic and deterministic epidemic models. Some parts of this project are hidden for confidentiality reasons, notably the prediction and decision making part on possible health policies to be conducted. This project was carried out as part of the study of the spread of the COVID-19 epidemic in France. 

