import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import STL

# -*- coding: utf-8 -*-
"""
Author: Daniel Felberg
Date: 2024-09-22
Version: 1.0
"""


def Cal_autocorrelation(data, max_lag, title):
    def autocorrelation(data, max_lag):
        n = len(data)
        mean = np.mean(data)
        numerator = sum((data[i] - mean) * (data[i - max_lag] - mean) for i in range(max_lag, n))
        denominator = sum((data[i] - mean) ** 2 for i in range(n))
        ry = numerator / denominator if denominator != 0 else 1.0
        return ry

    acf_values = [autocorrelation(data, lag) for lag in range(max_lag + 1)]

# Plot the ACF
    a = acf_values
    b = a[::-1]
    c =  b + a[1:]
    plt.figure()
    x_values = range(-max_lag, max_lag + 1)
    (markers, stemlines, baseline) = plt.stem(x_values, c, markerfmt = 'o')
    plt.setp(markers, color = 'red')
    m = 1.96/np.sqrt(len(data))
    plt.axhspan(-m,m, alpha = 0.2, color = 'blue')
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title(title)
    plt.show()


def ACF_PACF_Plot(y,lags,suptitle):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    fig.suptitle(suptitle)
    plt.subplot(211)
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.subplots_adjust(top=0.85)
    plt.show()
    
def ADF_Cal(x):
    x = x.interpolate()
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def kpss_test(timeseries):
    timeseries = timeseries.interpolate()
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
            kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)
    
def Cal_rolling_mean_var(value, y):
    fig, axs = plt.subplots(2, 1)

    rolling_means = []
    rolling_variances = []

    for i in range(1, rol_mean_var_len + 1):
        rolling_mean = value.head(i).mean()
        rolling_variance = value.head(i).var()

        rolling_means.append(rolling_mean)
        rolling_variances.append(rolling_variance)

    axs[0].plot(range(1, rol_mean_var_len + 1), rolling_means, label='Rolling Mean')
    axs[0].set_title(f'Rolling Mean - {y}')
    axs[0].set_xlabel('Samples')
    axs[0].set_ylabel('Magnitude')
    axs[0].legend()

    axs[1].plot(range(1, rol_mean_var_len + 1), rolling_variances, label='Rolling Variance')
    axs[1].set_title(f'Rolling Variance - {y}')
    axs[1].set_xlabel('Samples')
    axs[1].set_ylabel('Magnitude')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    
    
def differencing(data): # First Order Differencing
    diff_data = []
    for i in range(len(data)):
        if i == 0:
            diff_data.append(np.nan)
        else:
            diff_data.append((data[i]) - data[i-1])
    return diff_data

def str_trend_seasonal(T, S, R):
    F = np.maximum(0, 1 - (np.var(np.array(R)) / np.var(np.array(T + R))))
    print(f'Trend strength is {100 * F: .3f}%')  # returns percentage
    FS = np.maximum(0, 1 - (np.var(np.array(R)) / np.var(np.array(S + R))))
    print(f'Seasonality strength is {100 * FS: .3f}%')  # returns percentage
    
