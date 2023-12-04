import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse

import tensorflow as tf
from tensorflow import keras as ks

def EleccionDatos(data, Busqueda):
    data = pd.read_csv(data)
    data.set_index("2", inplace= True)
    Palabra = [cadena for cadena in data.columns if Busqueda in cadena]
    print(Palabra)
    return data