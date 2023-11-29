import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import warnings
import itertools
from statsmodels.stats.diagnostic import acorr_ljungbox

def EleccionDatos(data, Busqueda):
    data = pd.read_csv(data)
    data.set_index("2", inplace= True)
    Palabra = [cadena for cadena in data.columns if Busqueda in cadena]
    print(Palabra)
    return data[Palabra].dropna(axis = 0)


def VariableTotal(data):
    data = pd.DataFrame({
        'variable': data
    })
    data['logaritmico'] = np.log(data['variable'])
    data['diferencial'] = data['logaritmico'].diff(1)
    data.dropna(axis = 0 , inplace=	 True)
    return data

def Total_graphs(data):
    plt.figure(figsize = (15,6))
    col = data.columns
    for i in range(len(col)):
        plt.subplot(1,len(col), i+1)
        plt.plot(data[col[i]])
        plt.axhline( y = data[col[i]].mean(), color = 'red')
        plt.title(f'{col[i]}')
        plt.grid()

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

def Regresiones(data, t = 1):
    dt = pd.DataFrame({'variable':data})
    for i in range(1,t+1):
        dt[f'variable {i}'] = dt['variable'].diff(i)
    dt.dropna(axis = 0, inplace = True)
    model = sm.OLS(dt['variable'], dt.drop('variable', axis =  1)).fit()
    print(model.summary())


def ModeloARMA(data, p, q, trend = None):
    warnings.filterwarnings('ignore')
    order = (p, 0, q)  
    model = ARIMA(data, order=order, trend = trend)
    results = model.fit()
    pval = pd.DataFrame(results.summary().tables[1].data[1:], columns=results.summary().tables[1].data[0])
    tabla_coeficientes = pval.applymap(lambda x: x if not isinstance(x, str) else pd.to_numeric(x, errors='ignore'))
    pval = tabla_coeficientes[tabla_coeficientes.columns.values[4]]
    warnings.resetwarnings()
    return results, pval

def Pruebatoolkit(data, value ,trend = None, p_value = 0.05):
    warnings.filterwarnings('ignore')
    p = q = range(0, value)
    pdq = list(itertools.product(p, q))
    best_aic = float('inf')
    best_order = None
    for order in pdq:

        try:
            er = 0
            resultados, yval = ModeloARMA(data,order[0], order[1], trend= trend)
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

def Pormanteau_test(data, p , q, rezagos = 10):
    resultados, _ = ModeloARMA(data, p , q)
    residuos = resultados.resid
    q_stat, p_value = acorr_ljungbox(residuos, rezagos)
    resultados_df = pd.DataFrame({'Lag': np.arange(1, rezagos + 1), 'Q-Stat': q_stat, 'P-Valor': p_value})
    confianza = 1.96  # para un intervalo de confianza del 95%
    lower_bound = -confianza * residuos.std()
    upper_bound = confianza * residuos.std()

    # Grafica los residuos con intervalos de confianza
    plt.figure(figsize=(10, 6))
    plt.plot(residuos, marker='o', linestyle='', color='b', label='Residuos')
    plt.axhline(y=0, color='r', linestyle='--', label='Línea base')
    plt.axhline(y=lower_bound, color='g', linestyle='--', label='Intervalo de confianza (95%)')
    plt.axhline(y=upper_bound, color='g', linestyle='--')
    plt.title('Gráfico de Residuos con Intervalos de Confianza')
    plt.xlabel('Observación')
    plt.ylabel('Residuos')
    plt.legend()
    plt.show()