import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse
from itertools import permutations, product
from statsmodels.tsa.vector_ar.vecm import coint_johansen

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


def Johansen_Test(data, det_order, k_ar_diff, conf = 1):
    result = coint_johansen(data, det_order= det_order, k_ar_diff= k_ar_diff)
    a95 = result.cvm[conf,:]
    eig =result.lr2
    print('=================================================')
    print('Matriz de Valores Criticos vs Maximo Valor Propio')
    print('=================================================')
    print(f'            cvm     eigv        Hipotesis')
    for i in range(len(data.columns)):
        if eig[i] > a95[i]:
            sr = f'Se rechaza r = {i}, existe mas de {i} relación/nes de cointegración'
        else:
            sr = f'No hay suficiente evidencia para rechazar la hipótesis nula de r = {i}.'
        print(f'r = {i}     {a95[i]}    {round(eig[i], 4)}    {sr}')
    a95 = result.cvt[conf,:]
    eig =result.lr1
    print('=================================================')
    print('Tabla de Valores Criticos vs Traza Estadistica')
    print('=================================================')
    print(f'            cvt     TS          Hipotesis')
    for i in range(len(data.columns)):
        if eig[i] > a95[i]:
            sr = f'Se rechaza r = {i}, existe mas de {i} relación/nes de cointegración'
        else:
            sr = f'No hay suficiente evidencia para rechazar la hipótesis nula de r = {i}.'
        print(f'r = {i}     {a95[i]}    {round(eig[i], 4)}    {sr}')