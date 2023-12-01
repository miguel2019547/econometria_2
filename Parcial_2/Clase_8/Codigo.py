import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse

df = pd.read_csv('tus_datos.csv').dropna(axis = 0)
for column in df.columns:
    result = adfuller(df[column])
    print('================================================')
    print(f'ADF Statistic for {column}: {result[0]}')
    print(f'p-value for {column}: {result[1]}')
    print(f'Critical Values for {column}: {result[2]}')
    print('================================================')
    print('\n')

df_diff = df.diff().dropna()

model = VAR(df_diff)
results = model.fit()

# Resumen del modelo
print(results.summary())

# Predecir valores futuros
lag_order = results.k_ar
forecast = results.forecast(df_diff.values[-lag_order:], steps=10)  # Cambia 10 por el número de pasos que deseas predecir

# Convertir las predicciones a un DataFrame
forecast_df = pd.DataFrame(forecast, columns=df.columns)

# Calcular el error RMSE entre las predicciones y los valores reales
rmse_value = rmse(forecast_df, df_diff[-10:])  # Cambia 10 por el número de pasos que estás prediciendo
print(f'RMSE: {rmse_value}')
