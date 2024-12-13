# -*- coding: utf-8 -*-
"""
Author: Daniel Felberg
Date: 2024-10-14
Version: 1.0
"""

import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def LSTM_no_headlines(values):
    values = np.array(values)

    scaler = MinMaxScaler(feature_range=(0, 1))
    values_scaled = scaler.fit_transform(values.reshape(-1, 1))

    def create_sequences(data, seq_length):
        X = []
        y = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    seq_length = 1
    X, y = create_sequences(values_scaled, seq_length)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=25, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(tf.keras.layers.LSTM(units=25, return_sequences=False))
    model.add(tf.keras.layers.Dense(units=25))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions))

    return rmse,y_test,y_test_unscaled,predictions


def LSTM_headlines(stock_values,headlines,most_frequent_date_count):
    stock_df = stock_values.to_frame(name='Close')
    stock_df.index.name = 'Date'
    headlines.index.name = 'Date'

    headlines_grouped = headlines.groupby(headlines.index)['numeric_labels'].apply(list).reset_index()
    headlines_grouped.set_index('Date', inplace=True)
    combined_df = headlines_grouped.join(stock_df, how='outer')

    combined_noNA = combined_df.dropna(subset=['Close'])
    combined_noNA['numeric_labels'] = combined_noNA['numeric_labels'].apply(
        lambda x: [0] if isinstance(x, float) and pd.isna(x) else x)

    # change padding here (more padding = more 0s (less noise))
    combined_noNA['numeric_labels_padded'] = list(
        pad_sequences(combined_noNA['numeric_labels'], maxlen=most_frequent_date_count, padding='post', value=0)
    )

    time_steps = 1
    X_close = []
    X_numeric_labels = []
    y_close = []

    for i in range(time_steps, len(combined_noNA)):
        X_close.append(combined_noNA['Close'].iloc[i - time_steps:i].values)
        X_numeric_labels.append(np.array(combined_noNA['numeric_labels_padded'].iloc[i - time_steps:i].tolist()))
        y_close.append(combined_noNA['Close'].iloc[i])

    X_close = np.expand_dims(X_close, axis=2)  # Add a third dimension to X_close
    X_numeric_labels = np.array(X_numeric_labels)

    X_numeric_labels = np.mean(X_numeric_labels, axis=2, keepdims=True)  # combining using mean

    y_close = np.array(y_close)  # Shape: (samples,)

    X_combined = np.concatenate([X_close, X_numeric_labels], axis=2)

    split_index = int(len(X_combined) * 0.8)
    X_train, X_test = X_combined[:split_index], X_combined[split_index:]
    y_train, y_test = y_close[:split_index], y_close[split_index:]

    scaler = MinMaxScaler(feature_range=(0, 1))

    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
    X_train_scaled = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test_scaled = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=25, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
    model.add(tf.keras.layers.LSTM(units=25, return_sequences=False))
    model.add(tf.keras.layers.Dense(units=25))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train_scaled, y_train_scaled, epochs=20, batch_size=32, validation_data=(X_test_scaled, y_test_scaled))
    predictions_scaled = model.predict(X_test_scaled)
    predictions = scaler.inverse_transform(predictions_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return rmse, y_train,y_test,predictions