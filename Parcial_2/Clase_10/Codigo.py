import pandas as pd
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