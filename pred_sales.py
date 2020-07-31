##Test Script Qlik Sense

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta

###PATH SALES
PATH_SALES="C:/Users/jhonp/Desktop/qlik_py_test/VentasR_Python/DatosPythonIniciales/VentasPythonInciales.csv"

sales=pd.read_csv(PATH_SALES,sep=';',decimal=",")
sales['Fecha']=pd.to_datetime(sales['Fecha'])
sales=sales[['Ventas','Fecha']]


###convert to date YYYY-MM

sales['Fecha']=sales.apply(lambda x: x['Fecha'].strftime("%Y-%m"),axis=1)
sales=sales.groupby('Fecha').agg(vt_prom=('Ventas','mean')).reset_index()
sales.head()

sales=sales.sort_values('Fecha').set_index('Fecha').dropna()
sales=sales.loc['2005-03'::,]
sales.plot(figsize=(16,6))
plt.title("Sales Over time");

sales_agg=pd.Series(sales.vt_prom,pd.date_range(start='2005-03-01', periods=sales.shape[0], freq='MS'))

model = ExponentialSmoothing(sales_agg, trend="add", seasonal="add", seasonal_periods=11)
fit = model.fit()
pred = fit.forecast(11)

##write to csv
fecha=np.concatenate((sales_agg.index,pred.index))
values=np.concatenate((sales_agg.values,pred.values))
is_forecast=np.concatenate((np.repeat(0,sales_agg.shape[0]),np.repeat(1,pred.shape[0])))
dataout=pd.DataFrame({'fecha':fecha,'ventas_prom':values,'pronostico':is_forecast})
dataout.to_csv("C:/Users/jhonp/desktop/forecast.csv", sep='\t', encoding='utf-8',header=True,index=False,decimal=".")