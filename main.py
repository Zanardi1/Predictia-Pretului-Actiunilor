import warnings
from datetime import datetime
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def invboxcox(y, lmbda):
    if lmbda == 0:
        return np.exp(y)
    else:
        return np.exp(np.log(lmbda * y + 1) / lmbda)


data = pd.read_csv('Data.csv')

data.Timestamp = pd.to_datetime(data.Timestamp, unit='s')

data.index = data.Timestamp
data = data.resample("D").mean()
data_month = data.resample("M").mean()
data_year = data.resample("A-DEC").mean()
data_Q = data.resample("Q-DEC").mean()

fig = plt.figure(figsize=[15, 7])
plt.suptitle('Bitcoin exchanges, mean USD', fontsize=22)

plt.subplot(221)
plt.plot(data.Weighted_Price, '-', label='By Days')
plt.legend()

plt.subplot(222)
plt.plot(data_month.Weighted_Price, '-', label='By Months')
plt.legend()

plt.subplot(223)
plt.plot(data_Q.Weighted_Price, '-', label='By Quarters')
plt.legend()

plt.subplot(224)
plt.plot(data_year.Weighted_Price, '-', label='By Years')
plt.legend()

plt.show()

plt.figure(figsize=[15, 7])
sm.tsa.seasonal_decompose(data_month.Weighted_Price).plot()
print('Dickey - Fuller Test: p=%f' % sm.tsa.stattools.adfuller(data_month.Weighted_Price)[1])
plt.show()

data_month['Weighted_Price_box'], lmbda = stats.boxcox(data_month.Weighted_Price)
print('Dickey - Fuller Test: p=%f' % sm.tsa.stattools.adfuller(data_month.Weighted_Price)[1])

data_month['prices_box_diff'] = data_month.Weighted_Price_box - data_month.Weighted_Price_box.shift(12)
print('Dickey - Fuller Test: p=%f' % sm.tsa.stattools.adfuller(data_month.prices_box_diff[12:])[1])

data_month['prices_box_diff2'] = data_month.prices_box_diff - data_month.prices_box_diff.shift(1)
plt.figure(figsize=(15, 7))

sm.tsa.seasonal_decompose(data_month.prices_box_diff2[13:]).plot()
print('Dickey - Fuller test: p=%f' % sm.tsa.stattools.adfuller(data_month.prices_box_diff2[13:])[1])
plt.show()

data_month['prices_box_diff2'] = data_month.prices_box_diff - data_month.prices_box_diff.shift(1)
plt.figure(figsize=(15, 7))

sm.tsa.seasonal_decompose(data_month.prices_box_diff2[13:]).plot()
print('Dickey - Fuller test: p=%f' % sm.tsa.stattools.adfuller(data_month.prices_box_diff2[13:])[1])
plt.show()

plt.figure(figsize=(15, 7))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(data_month.prices_box_diff2[13:].values.squeeze(), lags=48, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(data_month.prices_box_diff2[13:].values.squeeze(), lags=48, ax=ax)
plt.tight_layout()
plt.show()

Qs = range(0, 2)
qs = range(0, 3)
Ps = range(0, 3)
ps = range(0, 3)
D = 1
d = 1
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
print(len(parameters_list))

results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')
for param in parameters_list:
    try:
        model = sm.tsa.statespace.SARIMAX(data_month.Weighted_Price_box, order=(param[0], d, param[1]),
                                          seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)
    except ValueError:
        print('Wrong parameters ', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])

result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by='aic', ascending=True).head())
print(best_model.summary())

plt.figure(figsize=(15, 7))
plt.subplot(211)
best_model.resid[13:].plot()
plt.ylabel('Residuals')
ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48, ax=ax)
print('Dickey - Fuller test: p=%f' % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])
plt.tight_layout
plt.show()

data_month2 = data_month[['Weighted_Price']]
date_list = [datetime(2017, 6, 30), datetime(2017, 7, 31), datetime(2017, 8, 31), datetime(2017, 9, 30),
             datetime(2017, 10, 31), datetime(2017, 11, 30), datetime(2017, 12, 31), datetime(2018, 1, 31),
             datetime(2018, 2, 28)]
future = pd.DataFrame(index=date_list, columns=data_month.columns)
data_month2 = pd.concat([data_month2, future])
data_month2['forecast'] = invboxcox(best_model.predict(start=0, end=75), lmbda)
plt.figure(figsize=(15, 7))
data_month2.Weighted_Price.plot()
data_month2.forecast.plot(color='r', ls='--', label='Predicted Weighted Price')
plt.legend()
plt.title('Bitcoin Exchanges, by months')
plt.ylabel('Mean USD')
plt.show()
