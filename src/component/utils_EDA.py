# -*- coding: utf-8 -*-
"""
Author: Daniel Felberg
Date: 2023-11-18
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt

def LSTM_viz(data,train_size, y_test_unscaled,predictions,ticker_symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[:train_size], data.values[:train_size], label='Train Data')
    plt.plot(data.index[train_size:train_size + len(y_test_unscaled)], y_test_unscaled, label='Test Data')
    plt.plot(data.index[train_size:train_size + len(predictions)], predictions, label='Predicted Data')
    plt.xlabel('Date')
    plt.ylabel(f'Stock Price ({ticker_symbol})')
    plt.legend()
    plt.title('LSTM Model - Actual vs Predicted')
    plt.show()

def LSTM_headlines_viz(y_train,y_test,predictions,ticker_symbol):
    train_size = len(y_train)
    test_size = len(y_test)
    train_data = np.concatenate([y_train, np.full_like(y_test, np.nan)])  # Combine y_train and NaN for the test portion
    test_data = np.concatenate([np.full_like(y_train, np.nan), y_test])  # Combine NaN for the train portion and y_test

    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label='Train Data')
    plt.plot(test_data, label='Test Data')
    plt.plot(np.arange(train_size, train_size + test_size), predictions, label='Predicted Data')
    plt.title('LSTM Model (Headlines) - Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel(f'Stock Price ({ticker_symbol})')
    plt.legend()
    plt.show()
