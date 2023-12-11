import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse
from itertools import permutations, product
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings
import itertools

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


def acf_ma(data, inx):
    acf_result = acf(data, qstat=True)[inx]
    t = len(acf_result)
    plt.stem(np.arange(t), acf_result)
    plt.title('Autocorrelation Function (MA)')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')

def pacf_ar(data):
    pacf_result = pacf(data, method='ols')
    t = len(pacf_result)
    plt.stem(np.arange(t), pacf_result)
    plt.title('Partial Autocorrelation Function (AR)')
    plt.xlabel('Order of AR')
    plt.ylabel('Partial Autocorrelation')


def DFuller(datos,steps = None):
    result = adfuller(datos,steps)

    # Extrae los resultados
    adf_statistic = result[0]
    p_value = result[1]

    # Imprime los resultados
    print(f'Estadístico ADF: {adf_statistic}')
    print(f'Valor p: {p_value}')
    # Compara el valor p con un umbral (por ejemplo, 0.05) para tomar decisiones
    if p_value <= 0.05:
        print("Rechazamos la hipótesis nula; la serie es estacionaria.")
    else:
        print("No podemos rechazar la hipótesis nula; la serie no es estacionaria.")

def Pruebatoolkit(data, value ,trend = None, p_value = 0.05):
    warnings.filterwarnings('ignore')
    p = d = q = range(0, value)
    pdq = list(itertools.product(p, d, q))
    best_aic = float('inf')
    best_order = None
    for order in pdq:

        try:
            er = 0
            resultados, yval = ModeloARIMA(data,order=order, trend= trend)
            for a in yval:
                if a > p_value:
                    er = 1
            if er == 0:             
                aic = resultados.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = order
        except:
            continue

    print(f"Mejor orden encontrado: {best_order} con AIC: {best_aic}")
    warnings.resetwarnings()
    return best_order

def ModeloARIMA(data, order, trend = None):
    warnings.filterwarnings('ignore')
    model = ARIMA(data, order=order, trend = trend)
    results = model.fit()
    pval = pd.DataFrame(results.summary().tables[1].data[1:], columns=results.summary().tables[1].data[0])
    tabla_coeficientes = pval.applymap(lambda x: x if not isinstance(x, str) else pd.to_numeric(x, errors='ignore'))
    pval = tabla_coeficientes[tabla_coeficientes.columns.values[4]]
    warnings.resetwarnings()
    return results, pval

def Columnas(data, Busqueda):
    Palabra = [cadena for cadena in data.columns if Busqueda in cadena]
    for a in Palabra:
        print(a)