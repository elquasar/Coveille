import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

file = "data.csv"
data = pd.read_csv(file, sep=";", parse_dates=[
                    'jour'], date_parser=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))

data = data[data.cl_age90 == 0]
data = data[data.dep == "27"]
print(data)

N = data["pop"].iloc[0]
print(N)


# moyenne sur semaine glissante, 

data.P = data.P.rolling(7, min_periods=1).mean()

I = data[["jour", "P"]]
I = I.reset_index(drop=True)
print(I)
plt.plot(data['jour'],data['P'])
plt.show()

# décès dans le département : 

file = "data_hsp.csv"
data = pd.read_csv(file, sep=";", parse_dates=[
                    'jour'], date_parser=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))