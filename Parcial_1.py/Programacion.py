import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
from acf import acf_plot
import warnings

class ModeloArima():
    def __init__(self, Datos):
        self.datos = Datos

    def Graficos(self):

        plt.plot(self.datos)
        plt.title('Serie Temporal')
        plt.grid()
        plt.show()
        acf_plot(self.datos,inx = 2)
        plt.show()
        plot_pacf(self.datos)
        plt.title('Autocorrelación Function (AR)')
        plt.show()




    def ModeloARMA(self,p,q):

        order = (p, 0, q)  
        model = ARIMA(self.datos, order=order)
        self.results = model.fit()
        self.summary = self.results.summary()


    def Prediccion(self):
        forecast_steps = len(self.datos) 
        self.forecast = self.results.get_forecast(steps=forecast_steps)
        forecast_index = range(len(self.datos), len(self.datos) + forecast_steps)


        plt.plot(self.datos, label='Observado')
        plt.plot(self.datos.index, self.forecast.predicted_mean, color='red', label='Predicción')
        plt.title('Serie Temporal con Predicciones')
        plt.legend()
        plt.show()



    def Pruebatoolkit(self, value):
        warnings.filterwarnings('ignore')
        p = d = q = range(0, value)
        pdq = list(itertools.product(p, d, q))
        best_aic = float('inf')
        best_order = None
        for order in pdq:
            try:
                model = ARIMA(self.datos, order=order)
                results = model.fit()
                aic = results.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = order
            except:
                continue

        print(f"Mejor orden encontrado: {best_order} con AIC: {best_aic}")
        self.corr = best_aic
        warnings.resetwarnings()
