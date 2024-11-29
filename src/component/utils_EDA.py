# -*- coding: utf-8 -*-
"""
Author: Daniel Felberg
Date: 2023-11-18
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt

def PlotModel(y_pred, X_train, X_test, Y_train, Y_test):
    train_indices = np.arange(len(X_train))
    test_indices = np.arange(len(X_train), len(X_train) + len(X_test))
    pred_indices = np.arange(len(X_train) + len(X_test), len(X_train) + len(X_test) + len(y_pred))

    plt.figure(figsize=(10, 6))
    plt.plot(train_indices, Y_train, label='Train Data', color='blue')
    plt.plot(test_indices, Y_test, label='Test Data', color='red')
    plt.plot(test_indices, y_pred, label='Predicted Values', color='green')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    plt.title('auto_dataset - Train, Test, and Predicted Values')
    plt.show()


# create plotting function
def ARIMA_viz(stock_values,yt_stock,yf_stock,predictions,order):
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(len(yt_stock)), yt_stock, label='Train Data')
    plt.plot(np.arange(len(yt_stock), len(stock_values)-1), yf_stock, label='Test Data')
    plt.plot(np.arange(len(yt_stock), len(stock_values)-1), predictions.tolist(), label='Predictions')
    # plt.ylim(-1, 1) # enable to see variance in predicted values
    plt.title(f'{order} Model Predictions vs Actual Stock Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Value')
    plt.legend()
    plt.show()

def LSTM_viz(data,train_size, y_test_unscaled,predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[:train_size], data.values[:train_size], label='Train Data')
    plt.plot(data.index[train_size:train_size + len(y_test_unscaled)], y_test_unscaled, label='Test Data')
    plt.plot(data.index[train_size:train_size + len(predictions)], predictions, label='Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.title('LSTM Model - Actual vs Predicted')
    plt.show()

def SARIMAX_viz(train_data,test_data,res,predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data, label='Training Data')
    plt.plot(test_data.index, test_data, label='Actual Test Data')
    plt.plot(test_data.index, predictions, label='Predicted Test Data')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('SARIMAX Model - Actual vs Predicted Stock Prices')
    plt.legend()
    plt.show()

    return res, predictions

def S2S_graph(real_stock_price,predicted_stock_price):
    plt.figure(figsize=(10, 6))
    plt.plot(real_stock_price, label='Real Stock Price')
    plt.plot(predicted_stock_price, label='Predicted Stock Price', linestyle='dashed')
    plt.title('Seq2Seq LSTM Model: Real vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

#%% CODE BELOW IS UNFINISHED SCRATCH CODE
#%% ARX model (label encoder)
# #
# from sklearn.preprocessing import LabelEncoder
#
# label_encoder = LabelEncoder()
# df_joined_clean['Label_numeric'] = label_encoder.fit_transform(df_joined_clean['Label'])
#
# ARX_yt_stock, ARX_yf_stock = train_test_split(df_joined_clean, shuffle=False, test_size=0.2)
#
# yt_endog = ARX_yt_stock['Close']
# yt_exog = ARX_yt_stock[['Label_numeric']]
# yf_endog = ARX_yf_stock['Close']
# yf_exog = ARX_yf_stock[['Label_numeric']]
#
# model_ARX = AutoReg(yt_endog, lags=1, exog=yt_exog)
# model_ARX_fit = model_ARX.fit()
#
# exog_oos = yf_exog
#
# predictions_ARX = model_ARX_fit.predict(start=len(yt_endog), end=len(df_joined_clean) - 1, exog_oos=exog_oos)