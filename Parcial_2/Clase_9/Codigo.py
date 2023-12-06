import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse
from itertools import permutations, product
import tensorflow as tf
from tensorflow import keras as ks

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



def GrangeTest(data,resultados):
    a = range(len(data.columns))
    perp = list(permutations(a))
    Principal = []
    Secundaria = []
    p_valor = []
    perma = None
    for perm in perp:
        if perm[0] != perma:
            p = data.columns[perm[0]]
            s1 = data.columns[perm[1]]
            s2 = data.columns[perm[2]]
            pval1 = resultados.test_causality(p,s1, kind = 'f').pvalue
            pval2 = resultados.test_causality(p,s2, kind = 'f').pvalue
            pval3 = resultados.test_causality(p,[s1,s2], kind = 'f').pvalue
            Principal.append(p)
            Principal.append(p)
            Principal.append(p)
            Secundaria.append(s1)
            Secundaria.append(s2)
            Secundaria.append(F'[{s1},{s2}]')
            p_valor.append(pval1)
            p_valor.append(pval2)    
            p_valor.append(pval3)
            perma = perm[0]

    return pd.DataFrame({
        'Endogena': Principal,
        'Exogenas': Secundaria,
        'p_value' : p_valor
    })

def prediccionVAR(resultados,data,steps):
    forecast = pd.DataFrame(resultados.forecast(y = data.values, steps = steps), columns= data.columns)
    fi = data.index[-1]
    ff = fi + pd.DateOffset(years=steps-1)
    forecast.set_index(pd.date_range(start=fi, end=ff, freq='Y'), inplace= True)
    return forecast