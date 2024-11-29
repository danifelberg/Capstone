import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt

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


def Cal_rolling_mean_var(value, y, rol_mean_var_len):

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


def gpac_values(ry, j_val, k_val):
    den = np.array([ry[np.abs(j_val + k - i)] for k in range(k_val) for i in range(k_val)]).reshape(k_val, k_val)
    col = np.array([ry[j_val+i+1] for i in range(k_val)])
    num = np.concatenate((den[:, :-1], col.reshape(-1, 1)), axis=1)
    return np.inf if np.linalg.det(den) == 0 else round(np.linalg.det(num)/np.linalg.det(den), 10)

def GPAC_table(ry, j_val, k_val):
    gpac_arr = np.full((j_val, k_val), np.nan)
    for k in range(1, k_val):
        for j in range(j_val):
            gpac_arr[j][k] = gpac_values(ry, j, k)
    gpac_arr = np.delete(gpac_arr, 0, axis=1)
    df = pd.DataFrame(gpac_arr, columns=list(range(1, k_val)), index=list(range(j_val)))

    plt.figure()
    sns.heatmap(df, annot=True, fmt='0.3f', linewidths=.5)
    plt.title('Generalized Partial Autocorrelation (GPAC) Table')
    plt.tight_layout()
    plt.show()
    print(df)


# THE GRID SEARCH FUNCTIONS BELOW WERE TAKEN FROM https://machinelearningmastery.com/

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.8)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

def exponential_smoothing(data):
    smoothed_data = data.ewm(alpha=0.3).mean()
    detrended_data = data - smoothed_data
    return detrended_data

def rev_diff(value, forecast):
    rev_forecast = []
    for i in range(0, len(forecast)):
        value += forecast[i]
        rev_forecast.append(value)
    return rev_forecast

def grid_search(data,min_order,max_order): # code from Liang!

    mse_test = []
    mse_val = []
    final_order = min_order
    best_mse_test = float('inf')
    best_mse_val = float('inf')

    train_size = int(len(data) * 0.8)
    val_size = int(len(data) * 0.1)
    test_size = len(data) - (train_size + val_size)

    train = data[:train_size]
    val = data[train_size:train_size + val_size]
    test = data[train_size + val_size:]

    for order in range(min_order, max_order + 1):
        try:
            model_ar = ARIMA(train, order=(order, 1, 0))
            model_fit = model_ar.fit()

            forecast_val = model_fit.forecast(steps=len(val))
            current_mse_val = round(mean_squared_error(val, forecast_val), 4)
            mse_val.append((order, current_mse_val))

            forecast_test = model_fit.forecast(steps=len(test))
            current_mse_test = round(mean_squared_error(test, forecast_test), 4)
            mse_test.append((order, current_mse_test))

            if current_mse_test < best_mse_test and current_mse_val < best_mse_val:
                best_mse_test = current_mse_test
                best_mse_val = current_mse_val
                final_order = order

        except Exception as e:
            print(f"Error fitting ARIMA model with order {order}: {e}")
            continue

    return final_order, mse_test, mse_val

#%% CODE BELOW IS UNFINISHED SCRATCH CODE

# #%% Plotting Trend, Seasonality, and Residuals
#
# # STL Decomposition (stationary)
# stl = STL(stock_values) #ACF peaks every 24 lags
# res = stl.fit()
# plt.figure(figsize=(10, 8))
# fig = res.plot()
# for ax in fig.get_axes():
#     ax.tick_params(axis='x', labelsize=6.5) # Resize x-axis labels for all subplots
# plt.tight_layout()
# plt.show()
#
# #%% Calculating Trend and Seasonality Strength
#
# T = res.trend
# S = res.seasonal
# R = res.resid
#
# def str_trend_seasonal(T, S, R):
#     F = np.maximum(0, 1 - (np.var(np.array(R)) / np.var(np.array(T + R))))
#     print(f'Trend strength is {100 * F: .3f}%')  # returns percentage
#     FS = np.maximum(0, 1 - (np.var(np.array(R)) / np.var(np.array(S + R))))
#     print(f'Seasonality strength is {100 * FS: .3f}%')  # returns percentage
#
# str_trend_seasonal(T, S, R)

##%% checking for seasonality
# stock_decomposition = seasonal_decompose(stock_values, model='additive', period=14)
# stock_decomposition.plot()
# plt.show()