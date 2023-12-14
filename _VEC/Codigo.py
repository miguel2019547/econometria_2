import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse
from itertools import permutations, product

def Tachos(data):
    data = pd.read_csv(data)
    data.set_index("2", inplace= True)
    data = data.reset_index()
    data['fecha'] = data['2']
    data.drop('2', axis = 1, inplace= True)
    data['fecha'] = pd.to_datetime(data['fecha'], format='%Y')
    data['fecha'] = data['fecha'] + pd.DateOffset(days = 364)
    data.set_index('fecha', inplace= True)

    return data

def Columnas(data, Busqueda):
    Palabra = [cadena for cadena in data.columns if Busqueda in cadena]
    for a in Palabra:
        print(a)