"""
Ce fichier contient la définition de fonctions utiles pour l'analyse de données
par exemple, il contient un parser qui récupère les données de SPF et les transforme
en compartiment pour les modèles type SIRD
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os

from .constant import N
from .constant import DATE_FRANCE


############################################################################################################


############################################################################################################


############################################################################################################


############################################################################################################


def plot_template():
    pd.plotting.register_matplotlib_converters()
    d = DATE_FRANCE
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_xlabel("emps en jour")
    ax.axvspan(d.Confinement1.Debut, d.Confinement1.Fin,
               color='g', alpha=0.20, label='1er confinement')
    ax.axvspan(d.Confinement1.Fin, d.Confinement2.Debut,
               color='r', alpha=0.20, label='Déconfinement')
    ax.axvspan(d.Confinement2.Debut, d.Confinement2.Fin,
               color='g', alpha=0.20, label='2ème confinement')
    ax.axvspan(d.Confinement2.Fin, d.Confinement3.Debut,
               color='r', alpha=0.20, label='Déconfinement')
    ax.axvspan(d.Confinement3.Debut, d.Confinement3.Fin,
               color='c', alpha=0.4, label='3ème confinement')
    ax.axvspan(d.Feu18h.Debut, d.Feu18h.Fin, color='b',
               alpha=0.20, label='Couvre-feu 18h généralisé')
    ax.axvspan(d.Feu19h.Debut, d.Feu19h.Fin, color='b',
               alpha=0.3, label='Couvre-feu 19h généralisé')
    return fig, ax


def data_parser_SPF():

    url = "https://www.data.gouv.fr/fr/datasets/r/08c18e08-6780-452d-9b8c-ae244ad529b3"

    # parsing hospital
    H = pd.read_csv(url, sep=";", parse_dates=[
                    'jour'], date_parser=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    H = H[H.cl_age90 == 0]
    H = H.groupby(["jour"])[['dc', 'rad', 'rea', 'hosp']].sum().reset_index()
    H['dc_diff'] = H.dc.diff()

    # parsing cas positif

    url = "https://www.data.gouv.fr/fr/datasets/r/dd0de5d9-b5a5-4503-930a-7b08dc0adc7c"
    I = pd.read_csv(url, sep=";", parse_dates=[
                    'jour'], date_parser=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    I = I[I.cl_age90 == 0][['jour', 'P', 'T']]
    I = I.rename(columns={'P': 'cas_positif', 'T': 'nombre_test'})
    datefin_cas = I[-1:].jour.values[0]
    I['nombre_test_cum'] = I.nombre_test.cumsum()

    # parsing cas positif avant
    url = "https://drive.google.com/file/d/1m9bWTBbG1QiY8WfLOJzifCZ6b_hl9Fye/view?usp=sharing"
    file_id = url.split('/')[-2]
    dwn_url = 'https://drive.google.com/uc?id=' + file_id

    df_ante = pd.read_csv(dwn_url, sep=',', parse_dates=[
        'date'], date_parser=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    df_ante = df_ante[["date", "total_cas_confirmes"]]
    df_ante = df_ante.rename(
        columns={"total_cas_confirmes": "cas_positif_cum", "date": "jour"})
    df_ante["cas_positif"] = df_ante.cas_positif_cum.diff()
    df_ante = df_ante[df_ante.jour < datetime.datetime(2020, 5, 13)]

    I = pd.concat([df_ante, I], axis=0, ignore_index=True)
    I.cas_positif = I.cas_positif.rolling(7, min_periods=1).mean()
    I['cas_positif_cum'] = I.cas_positif.cumsum()
    I = I.fillna(0)
    H = H.fillna(0)
    # alignement des dataframe sur les données SIDEP (toujours un peu plus en retard)
    H = H[H.jour <= I[-1:].jour.item()]
    # alignement des dataframe sur le début des données, 18 mars 2021
    I = I[I.jour >= H[:1].jour.item()]
    # retirer les dates de I
    I = I.drop('jour', axis=1)
    I = I.reset_index(drop=True)

    # concaténation des données hospitalières et SIDEP
    HI = pd.concat([I, H], axis=1)

    # calcul nombre guéri

    # =====================================================================
    #                   Estimation nombre de guéris
    # =====================================================================

    """
    Principe : 

    On prend le nombre de mort le jour t, puis on regarde le nombre d'infectés a t-delta t
    et on calcul le nombre de guéris comme étant : G(t) = Infecté(t-delta t) - Mort(t)
    Par la suite, on prend la somme cumulée du nombre de guéris car un guéris 
    est supposé immunisé et ne pourra plus être susceptible
    """
    # loi normal pour lee temps de guérison

    mu, sigma = 10, np.sqrt(5)

    G = []

    for jours in HI.jour:
        # aspect aléatoire dans la guérison
        delta_t = 10 #int(np.random.normal(mu, sigma, 1)[0])
        delta = datetime.timedelta(delta_t)
        deltajour = jours - delta
        deltajour = datetime.datetime.strftime(deltajour, '%Y-%m-%d')

        mort_t = HI[HI.jour == jours]['dc_diff'].item()
        if not HI[HI.jour == deltajour]['cas_positif'].tolist():
            gueri_t = 0
        else:

            infected_t_10 = HI.loc[HI['jour'] ==
                                   deltajour]['cas_positif'].item()
            gueri_t = infected_t_10 - mort_t

        G.append({'jour': jours, 'G': gueri_t})

    G = pd.DataFrame(G)
    G['G_cumul'] = G['G'].cumsum()
    G = G.drop('jour', axis=1)
    HIG = pd.concat([HI, G], axis=1)

    url_vax = "https://www.data.gouv.fr/fr/datasets/r/b273cf3b-e9de-437c-af55-eda5979e92fc"

    df_vax = pd.read_csv(url_vax, sep=';', parse_dates=[
        'jour'], date_parser=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    # les gens full vacinned sont ceux qui ont reçu la deuxième dose (max d'efficacité)

    df_vax = df_vax[['vaccin', 'jour', 'n_cum_dose2']]
    grouped = df_vax.groupby("vaccin", as_index=False)
    grouped_index = grouped.apply(
        lambda x: x.reset_index(drop=True).reset_index())

    list_data = [data for vaccin, data in grouped]
    # ajouter les dates jusqu'a la première date de HIG
    start = HIG[:1]['jour'][0]

    end = list_data[0][:1].jour
    days = datetime.timedelta(1)
    end = end - days
    end = end.iloc[0]

    v1 = list_data[0]
    between = pd.date_range(start, end)

    for df in list_data:

        vaccin_type = df.vaccin.iloc[0]
        df = df.rename(columns={"n_cum_dose2": "vaccin{}".format(vaccin_type)})
        completed = []
        for dates in between:
            completed.append(
                {"jour": dates, "vaccin{}".format(vaccin_type): 0})
        completed = pd.DataFrame(completed)

        df = pd.concat([completed, df[['jour', "vaccin{}".format(
            vaccin_type)]]], axis=0, ignore_index=True)
        HIG = pd.concat([HIG, df["vaccin{}".format(vaccin_type)]], axis=1)
    data = HIG[HIG.jour <= datefin_cas]

    # DONNEES SUR LES VARIANTS DEBUT = lundi 25 janvier 2021 (2021,1,25)
    url = "https://www.data.gouv.fr/fr/datasets/r/c43d7f3f-c9f5-436b-9b26-728f80e0fd52"

    df = pd.read_csv(url, sep=";")

    # selection tout age puis nettoyage colonnes inutiles
    df = df[df.cl_age90 == 0].reset_index(drop=True).drop(
        ['cl_age90'], axis=1).drop(['fra'], axis=1)
    df = df.drop(['Nb_tests_PCR_TA_crible', 'Prc_tests_PCR_TA_crible',
                  'Nb_susp_501Y_V1', 'Nb_susp_501Y_V2_3', 'Nb_susp_IND', 'Nb_susp_ABS'], axis=1)
    # 9 charactères a garder dans la date, le reste supprimer
    df.semaine = df.semaine.apply(lambda x: x[0:10])
    df = df.rename(columns={'semaine': 'jour', 'Prc_susp_501Y_V1': 'Prc_UK',
                            'Prc_susp_501Y_V2_3': 'Prc_BR_ZA', 'Prc_susp_IND': 'Prc_other', 'Prc_susp_ABS': 'Prc_origin'})
    # formatter les dates
    df.jour = pd.to_datetime(df['jour'], format='%Y-%m-%d')
    

    # récupérer la lsite des columns
    
    last_day = df[-1:].jour.item()
    df = df.drop(['jour'], axis=1)
    mask = df.columns.tolist()
    data = pd.concat([data, df], axis=1)

    # on shift toutes les lignes des nouvelles colonnes jusqu'a la date max de df puis on fill avec la meme
    # valeur jusqu'a la fin et on complete avec des zeros le reste

    maximum_shift = data[data.jour==last_day].index[0]
    nb_shift = maximum_shift - len(df.index)
    data[mask] = data[mask].shift(nb_shift)

    debut_data_variant = datetime.datetime(2021,1,25)
    index_debut_variant = data[data.jour == debut_data_variant].index[0]
    

    data.loc[0:index_debut_variant,['Prc_origin']] = 100
    data.loc[maximum_shift:,mask] = data.loc[maximum_shift-1,mask].tolist()
    data = data.fillna(0)
    

    # calcul des nouveaux cas par variant, on suppose que le pourcentage de test PCR pour un variant
    # est égal au nouveaux cas positif par variant par jour
    data['cas_positif_origin'] = (data['cas_positif'] * data['Prc_origin'])/100
    data['cas_positif_UK'] = (data['cas_positif'] * data['Prc_UK'])/100
    data['cas_positif_BR_ZA'] = (data['cas_positif'] * data['Prc_BR_ZA'])/100
    data['cas_positif_other'] = (data['cas_positif'] * data['Prc_other'])/100
    print(data.columns)

    return data




############################################################################################################


############################################################################################################


############################################################################################################


############################################################################################################
def give_compartiment(option=None):
    """Renvoie les données de SPF parsé pour nourrir un modèle SIR

    Args:
        option (string, optional): Type de modèle que l'on utilise (SIR,SIRD,...). Defaults to None.

    Returns:    
        pandas.DataFrame: DataFrame avec pour colonne les jours, et les noms de compartiments
    """

    # créer un repertoire Data s'il n'existe pas encore
    if not os.path.exists("data"):
        os.makedirs("data")
    today = datetime.date.today()
    # rechercher dans le dossier Data/{today}.csv les données de SPF ajd

    if os.path.isfile(r'data/{}.csv'.format(today)):
        # si les données d'aujourd'hui sont présente les récupérer
        print("Les données de SPF existent déjà...")
        print("...récupération des données dans le dossier data")
        data = pd.read_csv(r"data/{}.csv".format(today), parse_dates=[
                           'jour'], date_parser=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    else:
        # sinon les créer avec le parser
        print("Retrieving data from SPF...")
        data = data_parser_SPF()
        data.to_csv(r"data/{}.csv".format(today), index=False)
        print("Finished!")
        print("Data saved in data directory")

    print("You chose : ", option)
    print("Parsing data...")
    option1 = option == 'SIS'
    option2 = option == 'SIR'
    option3 = option == 'SIRD'
    option4 = option == 'SIRDH'
    option5 = option == 'SIRDV'
    option6 = option == 'SIRDVH'
    option7 = option == 'SIRDVHR'
    print("Data ready to feed a", option, "model!")
    print("Please use fit() method from the model")

    if option1:
        SI = pd.DataFrame()
        SI['jour'] = data['jour']
        SI['I'] = (data['cas_positif_cum'] - data['dc'] - data['G_cumul']) / N
        SI['S'] = 1 - SI['I']
        return SI

    if option2:

        SIR = pd.DataFrame()
        SIR['jour'] = data['jour']
        SIR['R'] = data['G_cumul'] / N
        SIR['I'] = (data['cas_positif_cum'] - data['dc'] - data['G_cumul']) / N
        SIR['S'] = 1 - SIR['I'] - SIR['R']
        return SIR

    if option3:
        SIRD = pd.DataFrame()
        SIRD['jour'] = data['jour']
        SIRD['D'] = data['dc'] / N
        SIRD['R'] = data['G_cumul'] / N
        SIRD['I'] = data['cas_positif_cum'] / N - SIRD['R'] - SIRD['D']
        SIRD['S'] = 1 - SIRD['I'] - SIRD['R'] - SIRD['D']
        SIRD = SIRD[SIRD.columns[::-1]]
        return SIRD

    if option5:
        SIRDV = pd.DataFrame()
        SIRDV['jour'] = data['jour']
        SIRDV['V'] = data['vaccin0'] / N
        SIRDV['D'] = data['dc'] / N
        SIRDV['R'] = data['G_cumul'] / N
        SIRDV['I'] = data['cas_positif_cum'] / N - SIRDV['R'] - SIRDV['D']
        SIRDV['S'] = 1 - SIRDV['I'] - SIRDV['R'] - SIRDV['D'] - SIRDV['V']
        SIRDV = SIRDV[SIRDV.columns[::-1]]
        return SIRDV


############################################################################################################


############################################################################################################


############################################################################################################


############################################################################################################
