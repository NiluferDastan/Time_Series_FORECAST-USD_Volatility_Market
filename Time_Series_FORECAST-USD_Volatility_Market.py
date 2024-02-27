#!/usr/bin/env python
# coding: utf-8

# ## Time Series-FORECAST/USD Volatility Market(ARIMA-SARIMA-LSTM)

# In[ ]:


from evds import evdsAPI 
evds = evdsAPI('zp7N5mCZWP') 
evds.get_data(['TP.DK.USD.A.YTL','TP.DK.EUR.A.YTL'], startdate="01-01-2019", enddate="01-01-2023")


# In[ ]:


cat=evds.main_categories
cat


# In[ ]:


sub=evds.get_sub_categories(2)
sub


# In[ ]:


seri=evds.get_series("bie_dkdovytl")
seri


# In[ ]:


import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf,month_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm


# In[ ]:


buying= evds.get_data(["TP.DK.USD.A.YTL"],startdate="23-02-2018",enddate="23-02-2024",frequency=5,aggregation_types="avg")
selling= evds.get_data(["TP.DK.USD.S.YTL"],startdate="23-02-2018",enddate="23-02-2024",frequency=5,aggregation_types="avg")
buying


# In[ ]:


#Convert to date_time format into Year/Month/Day 


# Translate columns to English for understandability

# In[ ]:


buying_date=pd.date_range("23.01.2018", periods=len(buying["Tarih"]),freq="M")
buying["Tarih"]=buying_date
selling_date=pd.date_range("23.01.2018", periods=len(selling["Tarih"]),freq="M")
selling["Tarih"]=selling_date


# In[ ]:


buying


# In[ ]:


buying.rename(columns={"TP_DK_USD_A_YTL":"USD_Buying"},inplace=True)
selling.rename(columns={"TP_DK_USD_S_YTL":"USD_Selling"},inplace=True)
buying.rename(columns={"Tarih":"Date"},inplace=True)
selling.rename(columns={"Tarih":"Date"},inplace=True)
buying


# Convert to dates into index

# In[ ]:


buying.set_index("Date",inplace=True)
selling.set_index("Date",inplace=True)


# In[ ]:


buying1=buying.iloc[:-12]          #This dataset used to predict when building the model
lastyear_buying=buying.iloc[-12:]  #This dataset for the predictions we will make after building the model


# In[ ]:


lastyear_buying


# In[ ]:


usd_exchange_rate=pd.concat([buying,selling],axis=1)


# In[ ]:


usd_exchange_rate 


# In[ ]:


fig = plt.figure(figsize=(10,5))
usd_exchange_rate.USD_Buying.plot(label='USD Buying')
usd_exchange_rate.USD_Selling.plot(label='USD Selling')
plt.legend(loc='best')
plt.title('Yearly Exchange Rates', fontsize=12)
plt.legend()
plt.show()


# Son yıllar volatil bir piyasaya sahip olduğumuzda genelde üstel bir trend gösteriyor.
# Seri trend barındırdığı için durağan değildir. Anca mevsimsel bir bileşen var mı bunu otokorelasyon üzerinden bakıcaz.

# In[ ]:


ACF= plot_acf(buying, lags=20,zero=False)
plt.show(ACF)
plt.savefig('ACF.jpg', dpi=300,transparent=True)

PACF= plot_pacf(selling, lags=20,zero=False)
plt.show(PACF)
plt.savefig('PACF.jpg', dpi=300,transparent=True)


# Data Preparation:
# The original data needs to achieve stationary before fitting into any time series model, the most common method is to take the first order differentiation. The ADF test is used to test if the difference data achieve stationarity.

# In[ ]:


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


# In[ ]:


def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)


# ADF and KPSS test must apply on the data.
# 

# In[ ]:


adf_test(usd_exchange_rate["USD_Buying"])


# In[ ]:


def kpss_test(timeseries):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)


# In[ ]:


kpss_test(usd_exchange_rate["USD_Buying"])


# In[ ]:


with open('kpss_test', 'wb') as files:
    pickle.dump(kpss_test, files)


# #### ADF — Augmented Dickey — Fuller
# H_0= The process does not contains stationary.
# H_1: The times series is higly likely to lack a unit root and thus can be considered as associated with stationary data.
# Based upon the significance level of 0.05 and the p-value of ADF test=0.99, the null hypothesis can not be rejected. Hence, the series is non-stationary.
# 
# The KPSS tests gives the following results – test statistic, p value and the critical value at 1%, 5% , and 10% confidence intervals.
# 
# -------
# Based upon the significance level of 0.05 and the p-value of KPSS test, there is evidence for rejecting the null hypothesis in favor of the alternative. Hence, the series is non-stationary as per the KPSS test.
# 
# ----
# * It is always better to apply both the tests, so that it can be ensured that the series is truly stationary. Possible outcomes of applying these stationary tests are as follows:
# 
#     - *Case 1: Both tests conclude that the series is not stationary - The series is not stationary*
#     - *Case 2: Both tests conclude that the series is stationary - The series is stationary*
#     - *Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.*
#     - *Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary. The differenced series is checked for stationarity.*
# 
# Here, due to the difference in the results from ADF test and KPSS test, it can be inferred that the series is trend stationary and not strict stationary. The series can be detrended by differencing or by model fitting.**

# In[ ]:


#setting the split date
split_date = pd.Timestamp('30-04-2023')

# creating training dataframe 
USD_Price = buying.loc[:split_date]
# creating test dataframe 
Forecast = buying.loc[split_date:]

#plotting train test dataframe as aline plot
ax = USD_Price.plot(kind='line',figsize=(13,5))
Forecast.plot(ax=ax,kind='line',figsize=(13,5))
plt.legend(['USD_Price', 'Forecast'])


# In[ ]:


##For non-seasonal data
#p=1, d=1, q=0 or 1

from statsmodels.tsa.arima.model import ARIMA
model=ARIMA(usd_exchange_rate['USD_Buying'],order=(1,1,5))
model_fit=model.fit()
model_fit.summary()


# ### Test for stationarity:
# If the test statistic is less than the critical value, we can reject the null hypothesis (aka the series is stationary). When the test statistic is greater than the critical value, we fail to reject the null hypothesis (which means the series is not stationary).
# 
# In our above example, the test statistic > critical value, which implies that the series is not stationary. This confirms our original observation which we initially saw in the visual test.
# - AR(1), MA(1), MA(2) means they significantly predict in the model. But MA(3), MA(4), MA(5) are less than the critical value which means 3.lag 4.lag and 5.lag is not stationary.
# 

# In[ ]:


model=sm.tsa.statespace.SARIMAX(usd_exchange_rate['USD_Buying'],order=(1, 1, 5),seasonal_order=(1,1,1,12))
results=model.fit()
usd_exchange_rate['Forecast']=results.predict(start=20,end=74,dynamic=True)
usd_exchange_rate[['USD_Buying','Forecast']].plot(figsize=(12,8))


# In[ ]:


# make predictions
predictions = model_fit.predict(start=len(USD_Price), end=len(USD_Price)+len(Forecast)-1, dynamic=False)
for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], USD_Price[i]))
    rmse = sqrt(mean_squared_error(usd_exchange_rate["USD_Buying"], predictions))
print('Test RMSE: %.3f' % rmse)
# plot results
plt.plot(usd_exchange_rate["USD_Buying"])
plt.plot(predictions, color='red')
plt.show()

