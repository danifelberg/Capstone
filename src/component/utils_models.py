# -*- coding: utf-8 -*-
"""
Author: Daniel Felberg
Date: 2024-10-14
Version: 1.0
"""

from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import statsmodels.api as sm



def AR_model(y, lags):
    Y_train, Y_test = train_test_split(y, test_size=0.2, shuffle=False)
    model = AutoReg(Y_train, lags)
    model_fit = model.fit()
    y_pred = model_fit.predict(start=Y_train,end=len(y))
    forecast = model_fit.forecast(len(Y_test))
    return y_pred, forecast

def ARIMA_stock(y,order):
    Y_train, Y_test = train_test_split(y, test_size=0.2, shuffle=False)
    model = ARIMA(Y_train,order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(Y_test))
    mse = mean_squared_error(Y_test, predictions)
    rmse = np.sqrt(mse)

    return Y_train,Y_test,predictions,rmse

def SARIMA_stock(data,order,seasonal_order):
    mod = sm.tsa.statespace.SARIMAX(data,trend='c', order=order,
                                    seasonal_order=seasonal_order,
                                    simple_differencing=True)
    res = mod.fit(disp=False)
    return res

#%% CODE BELOW IS UNFINISHED SCRATCH CODE

# #%% ARX model dataviz
#
# plt.figure(figsize=(10,6))
# plt.plot(np.arange(len(yt_endog)), yt_endog, label='Train Data')
# plt.plot(np.arange(len(yt_endog), len(df_joined_clean)), yf_endog, label='Test Data')
# plt.plot(np.arange(len(yt_endog), len(df_joined_clean)), predictions_ARX, label='Predictions', linestyle='dashed')
# plt.title('ARX(1) Model Predictions vs Actual Stock Values')
# plt.xlabel('Time Steps')
# plt.ylabel('Stock Value')
# plt.legend()
# plt.show()
#
# mse_ARX = mean_squared_error(yf_endog, predictions_ARX)
# print(f"Mean Squared Error (ARX): {mse_ARX}") # higher MSE 98.3 > 13
#
# # print(model_ARX_fit.summary())
# #
# # #%% ARX residuals
# #
# # ACF_PACF_Plot(stock_df_clean['1st_order_diff'],lags=10,suptitle='ACF/PACF of Stock') #PACF shows ideal lag to be 2
# #
# # # residuals_ARX = model_ARX_fit.resid
# # # plot_acf(residuals_ARX)
# # # plt.tight_layout()
# # # plt.show()
# #
# # residuals_ARX = model_ARX_fit.resid
# # ACF_PACF_Plot(residuals_ARX, lags=10, suptitle='ACF/PACF of ARX Residuals')
#
# #%% ARX model evaluaton (AIC, BIC)
#
# print(f'AIC (ARX): {model_ARX_fit.aic}') # lower AIC (367.3 < 1786.2)
# print(f'BIC (ARX): {model_ARX_fit.bic}') # lower BIC (375.4 < 1797.6)
#
# #%% SEQ2SEQ
#
# from sklearn.preprocessing import MinMaxScaler
#
# scaler = MinMaxScaler(feature_range=(0, 1))
# df_scaled = scaler.fit_transform(df_joined_clean[['Close', 'Label_numeric']])
#
# # Create sequences (e.g., x days of data for prediction)
# def create_sequences(data, seq_length):
#     x, y = [], []
#     for i in range(len(data) - seq_length):
#         x.append(data[i:i + seq_length, :])  # First x days as input
#         y.append(data[i + seq_length, 0])    # x+1th day's stock price as target
#     return np.array(x), np.array(y)
#
# seq_length = 4  # Number of weeks to look back (4 = 1 month)
# X, y = create_sequences(df_scaled, seq_length)
#
# # Split into training and test sets
# train_size = int(len(X) * 0.8)
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]
#
# #%% Convert data to PyTorch tensors
#
# import torch
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.float32)
#
# #%% Defining the Seq2Seq Model in PyTorch
# import torch.nn as nn
#
# class Seq2SeqLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=1):
#         super(Seq2SeqLSTM, self).__init__()
#
#         # Define the LSTM layer
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#
#         # Define the output layer (linear layer)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         # Pass the input through the LSTM layer
#         lstm_out, _ = self.lstm(x)
#
#         # Use the last output of the LSTM for the final prediction
#         out = self.fc(lstm_out[:, -1, :])  # Take the output at the last time step
#         return out
#
# #%% define model parameters
# input_size = 2  # 'Close' and 'Label_numeric'
# hidden_size = 64  # Number of hidden units in the LSTM
# output_size = 1  # Predicting a single value (stock price)
# num_layers = 1  # Single layer LSTM
#
# #%%
# # Initialize the model
# model = Seq2SeqLSTM(input_size, hidden_size, output_size, num_layers)
#
# #%%
# # Loss function and optimizer
# criterion = nn.MSELoss()  # Mean Squared Error loss
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
#
# #%%
# # Training the model
# num_epochs = 50  # You can adjust this
# batch_size = 32  # You can also adjust this
# train_losses = []
#
# for epoch in range(num_epochs):
#     model.train()  # Set the model to training mode
#     optimizer.zero_grad()  # Zero the gradients
#
#     # Forward pass
#     output = model(X_train)
#
#     # Compute loss
#     loss = criterion(output.squeeze(), y_train)
#
#     # Backward pass and optimize
#     loss.backward()
#     optimizer.step()
#
#     train_losses.append(loss.item())
#
#     if (epoch+1) % 10 == 0:
#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
#
#
# #%% Set the model to evaluation mode
# model.eval()
#
# # Make predictions on the test set
# with torch.no_grad():
#     predictions = model(X_test)
#
# # Inverse scale the predicted stock prices
# predicted_stock_price = scaler.inverse_transform(np.concatenate((predictions.numpy(), np.zeros((predictions.shape[0], 1))), axis=1))[:, 0]
#
# # Inverse scale the actual test stock prices
# real_stock_price = scaler.inverse_transform(np.concatenate((y_test.numpy().reshape(-1, 1), np.zeros((y_test.shape[0], 1))), axis=1))[:, 0]
#
# plt.figure(figsize=(10, 6))
# plt.plot(real_stock_price, label='Real Stock Price')
# plt.plot(predicted_stock_price, label='Predicted Stock Price', linestyle='dashed')
# plt.title('Seq2Seq LSTM Model: Real vs Predicted Stock Prices')
# plt.xlabel('Time')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()
#
# #%% Plotting both vars against time
#
# # fig, ax1 = plt.subplots(figsize=(10, 6))
# #
# # ax1.plot(df_joined_clean['Close'])
# # ax2 = ax1.twinx()
# # ax2.plot(df_joined_clean['label_numeric'])
# # plt.show()
# #
# # plt.plot(df_joined_clean['Close'])
# # plt.plot(df_joined_clean['label_numeric'])
# # plt.show()

# #%% AR/ARX model --> USE THIS
#
# steps_ahead = 197
# out = f'AIC: {0:0.3f}, HQIC: {1:0.3f}, BIC: {2:0.3f}'
#
#
# model = AutoReg(stock_df_clean['1st_order_diff'], lags=[1]).fit()
# forecast = model.predict(start=len(yt_stock), end=len(yf_stock) + steps_ahead - 1)
# print(out.format(model.aic, model.hqic, model.bic))
#
# # res = AutoReg(stock_values, lags = [2]).fit()
# # print(out.format(res.aic, res.hqic, res.bic))
#
# model2 = AutoReg(endog=df_joined_clean['Close'], lags=[1],exog=df_joined_clean['label_numeric']).fit()