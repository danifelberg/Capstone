#%%

import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline
import torch
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#%% defining scraper function
# url = "https://www.ft.com/search?q=j%26j&page=2&sort=relevance&isFirstView=false"

# def scrape_ft_headlines():
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')
#
#     # Find all headlines
#     headlines = [headline.text.strip() for headline in soup.find_all('a', class_='js-teaser-heading-link')]
#
#     date_elements = soup.find_all('time')
#     dates = []
#
#     for date_elem in date_elements:
#         try:
#             # Extract the datetime attribute or the text content
#             date_str = date_elem['datetime'] if date_elem.has_attr('datetime') else date_elem.text
#             # Parse the date string into a datetime object
#             dates.append(pd.to_datetime(date_str).date())
#         except Exception as e:
#             print(f"Error parsing date: {e}")
#             dates.append(None)
#
#     if len(dates) < len(headlines):
#         # If fewer dates than headlines, pad dates with None
#         dates.extend([None] * (len(headlines) - len(dates)))
#     elif len(headlines) < len(dates):
#         # If more dates than headlines (unlikely but possible), truncate the dates
#         dates = dates[:len(headlines)]
#
#     return headlines, dates

# Define the function to scrape headlines
def scrape_ft_headlines(num_pages):
    all_headlines = []
    all_dates = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    for page_num in range(1, num_pages + 1):
        # Adjust the URL to the current page number
        url = f"https://www.ft.com/search?q=j%26j&page={page_num}&sort=relevance&isFirstView=false"
        response = requests.get(url, headers=headers)

        # Check if the response was successful
        print(f"Fetching page {page_num}: Status code {response.status_code}")

        if response.status_code != 200:
            print(f"Error fetching page {page_num}.")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')

        # Debugging: print the HTML to verify the structure
        # print(soup.prettify()) # --> optional

        # Find all headlines
        headlines = [headline.text.strip() for headline in soup.find_all('a', class_='js-teaser-heading-link')]

        # Find all dates
        date_elements = soup.find_all('time')
        dates = []

        for date_elem in date_elements:
            try:
                # Extract the datetime attribute or the text content
                date_str = date_elem['datetime'] if date_elem.has_attr('datetime') else date_elem.text
                # Parse the date string into a datetime object
                dates.append(pd.to_datetime(date_str).date())
            except Exception as e:
                print(f"Error parsing date: {e}")
                dates.append(None)

        # Handle cases where there are more headlines or dates
        if len(dates) < len(headlines):
            dates.extend([None] * (len(headlines) - len(dates)))
        elif len(headlines) < len(dates):
            dates = dates[:len(headlines)]

        # Add the current page's headlines and dates to the master list
        all_headlines.extend(headlines)
        all_dates.extend(dates)

    return all_headlines, all_dates

#%% running scraper

headlines, dates = scrape_ft_headlines(num_pages=42)

# Check if headlines and dates were successfully scraped
if headlines and dates:
    print(f"Number of headlines: {len(headlines)}")
    print(f"Number of dates: {len(dates)}")

    if len(headlines) != len(dates):
        print("Warning: Number of headlines and dates are not equal!")

    # Create a DataFrame with the headlines and corresponding dates
    df = pd.DataFrame({
        'Headline': headlines,
        'Date': dates
    })

    # Set 'Date' as the index of the DataFrame
    df.set_index('Date', inplace=True)

    # Sort the DataFrame by index
    df_sorted = df.sort_index()

    # Remove rows with null values in the index
    df_sorted_clean = df_sorted[df_sorted.index.notnull()]

    # Output the cleaned DataFrame
    print(df_sorted_clean.head())
else:
    print("Error: No headlines or dates returned. Please check the scrape_ft_headlines function.")

#%% performing sentiment analysis

pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

#%%
def sentiment_analysis(list_of_headlines):
    labels = []  # To store sentiment labels
    scores = []  # To store sentiment scores

    for headline in list_of_headlines:
        # Perform sentiment analysis on each headline
        result = pipe(headline)

        # Assuming the result contains a list of dictionaries with 'label' and 'score'
        label = result[0]['label']  # Extract the 'label'
        score = result[0]['score']  # Extract the 'score'

        # Append the label and score to respective lists
        labels.append(label)
        scores.append(score)

    # Return the lists of labels and scores
    return labels, scores

labels, scores = sentiment_analysis(df_sorted_clean['Headline'])

#%% adding labels and scores to df
df_sorted_clean['Label'] = labels
df_sorted_clean['Score'] = scores


#%% FOR NEXT WEEK:

# work on ARX model
# AR Model for J&J stock --> yahoo finance
# create time series class (AR model)
# compare AR (just J&J stock) with ARX (allows for exogenous inputs)

#%% date range

min_date = df_sorted_clean.index.min()
max_date = df_sorted_clean.index.max()
print(f'Min date: {min_date}')
print(f'Max date: {max_date}')

#%% fetching JNJ stock data from yahoo finance

company = 'JNJ' # change depending on stock you want to change

stock_data = yf.download(company, start=min_date, end=max_date, interval='1wk')
stock_values = stock_data['Close']

#%% dataviz of stock data

plt.plot(stock_values)
# plt.xticks(rotation=20)
plt.xlabel("Date")
plt.ylabel("Closing Stock Value")
plt.tight_layout()
plt.show()

# data appears to be trended (non-stationary)

#%% checking for equal sampling

time_diffs = stock_values.index.to_series().diff().dropna()
if time_diffs.nunique() == 1:
    print("The data is equally sampled.")
else:
    print("The data is not equally sampled.")
    print(time_diffs.value_counts())

# time_diffs.plot(title="Time Differences Between Consecutive Observations")
# plt.tight_layout()
# plt.show()

#%% ADF and KPSS
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

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

ADF_Cal(stock_values)
kpss_test(stock_values)

#%% rolling means and variance

rol_mean_var_len = len(stock_values)

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

Cal_rolling_mean_var(stock_values, "Stock")

#%% data is non-stationary, differencing is needed

def differencing(data):  # First Order Differencing
    diff_data = []
    for i in range(len(data)):
        if i == 0:
            diff_data.append(np.nan)  # First value will always be NaN due to differencing
        else:
            diff_data.append(data[i] - data[i - 1])
    return diff_data

stock_df = pd.DataFrame(stock_values, columns=['Close'])
stock_df['1st_order_diff'] = differencing(stock_df['Close'])

stock_df_clean = stock_df.dropna(subset=['1st_order_diff'])


#%% re-checking stationarity

ADF_Cal(stock_df_clean['1st_order_diff'])
kpss_test(stock_df_clean['1st_order_diff'])
Cal_rolling_mean_var(stock_df_clean['1st_order_diff'], "Stock")

# Data is now stationary --> 2nd order diff??
# overdifferencing!!!

#%% ACF and PACF

from statsmodels.graphics.tsaplots import plot_acf , plot_pacf

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

ACF_PACF_Plot(stock_df_clean['1st_order_diff'],lags=10,suptitle='ACF/PACF of Stock') #PACF shows ideal lag to be 2

#%% AR model train/test (stock data by itself)

yt_stock, yf_stock = train_test_split(stock_values, shuffle=False, test_size=0.2)

model = AutoReg(yt_stock, lags=1) # ACF/PACF plots suggest ideal lag to be 1
model_fit = model.fit()

predictions = model_fit.predict(start=len(yt_stock), end=len(stock_values) - 1)

# print(model_fit.summary())

#%% AR model dataviz

# create plotting function
plt.figure(figsize=(10,6))
plt.plot(np.arange(len(yt_stock)), yt_stock, label='Train Data')
plt.plot(np.arange(len(yt_stock), len(stock_values)), yf_stock, label='Test Data')
plt.plot(np.arange(len(yt_stock), len(stock_values)), predictions, label='Predictions', linestyle='dashed')
plt.title('AR(1) Model Predictions vs Actual Stock Values')
plt.xlabel('Time Steps')
plt.ylabel('Stock Value')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(yf_stock, predictions)
print(f"Mean Squared Error: {mse}")

# use non-stationary data, AIC+BIC were both lower!!!


#%% AR model residuals

residuals = model_fit.resid
plot_acf(residuals)
plt.tight_layout()
plt.show() # no autocorrelation exhibited

#%% model evaluaton (AIC, BIC)

print(f'AIC: {model_fit.aic}')
print(f'BIC: {model_fit.bic}')

#%% joining data using datetime

df_joined = df_sorted_clean.join(stock_df_clean)
df_joined_clean = df_joined.dropna()

#%% ARX model (label encoder)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df_joined_clean['Label_numeric'] = label_encoder.fit_transform(df_joined_clean['Label'])

ARX_yt_stock, ARX_yf_stock = train_test_split(df_joined_clean, shuffle=False, test_size=0.2)

yt_endog = ARX_yt_stock['Close']
yt_exog = ARX_yt_stock[['Label_numeric']]
yf_endog = ARX_yf_stock['Close']
yf_exog = ARX_yf_stock[['Label_numeric']]

model_ARX = AutoReg(yt_endog, lags=1, exog=yt_exog)
model_ARX_fit = model_ARX.fit()

exog_oos = yf_exog

predictions_ARX = model_ARX_fit.predict(start=len(yt_endog), end=len(df_joined_clean) - 1, exog_oos=exog_oos)

#%% ARX model dataviz

plt.figure(figsize=(10,6))
plt.plot(np.arange(len(yt_endog)), yt_endog, label='Train Data')
plt.plot(np.arange(len(yt_endog), len(df_joined_clean)), yf_endog, label='Test Data')
plt.plot(np.arange(len(yt_endog), len(df_joined_clean)), predictions_ARX, label='Predictions', linestyle='dashed')
plt.title('ARX(1) Model Predictions vs Actual Stock Values')
plt.xlabel('Time Steps')
plt.ylabel('Stock Value')
plt.legend()
plt.show()

mse_ARX = mean_squared_error(yf_endog, predictions_ARX)
print(f"Mean Squared Error (ARX): {mse_ARX}") # higher MSE 98.3 > 13

# print(model_ARX_fit.summary())

#%% ARX residuals

ACF_PACF_Plot(stock_df_clean['1st_order_diff'],lags=10,suptitle='ACF/PACF of Stock') #PACF shows ideal lag to be 2

# residuals_ARX = model_ARX_fit.resid
# plot_acf(residuals_ARX)
# plt.tight_layout()
# plt.show()

residuals_ARX = model_ARX_fit.resid
ACF_PACF_Plot(residuals_ARX, lags=10, suptitle='ACF/PACF of ARX Residuals')

#%% ARX model evaluaton (AIC, BIC)

print(f'AIC (ARX): {model_ARX_fit.aic}') # lower AIC (367.3 < 1786.2)
print(f'BIC (ARX): {model_ARX_fit.bic}') # lower BIC (375.4 < 1797.6)

#%% SEQ2SEQ

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df_joined_clean[['Close', 'Label_numeric']])

# Create sequences (e.g., x days of data for prediction)
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length, :])  # First x days as input
        y.append(data[i + seq_length, 0])    # x+1th day's stock price as target
    return np.array(x), np.array(y)

seq_length = 4  # Number of weeks to look back (4 = 1 month)
X, y = create_sequences(df_scaled, seq_length)

# Split into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

#%% Convert data to PyTorch tensors

import torch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

#%% Defining the Seq2Seq Model in PyTorch
import torch.nn as nn

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Seq2SeqLSTM, self).__init__()

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Define the output layer (linear layer)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Pass the input through the LSTM layer
        lstm_out, _ = self.lstm(x)

        # Use the last output of the LSTM for the final prediction
        out = self.fc(lstm_out[:, -1, :])  # Take the output at the last time step
        return out

#%% define model parameters
input_size = 2  # 'Close' and 'Label_numeric'
hidden_size = 64  # Number of hidden units in the LSTM
output_size = 1  # Predicting a single value (stock price)
num_layers = 1  # Single layer LSTM

#%%
# Initialize the model
model = Seq2SeqLSTM(input_size, hidden_size, output_size, num_layers)

#%%
# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

#%%
# Training the model
num_epochs = 50  # You can adjust this
batch_size = 32  # You can also adjust this
train_losses = []

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the gradients

    # Forward pass
    output = model(X_train)

    # Compute loss
    loss = criterion(output.squeeze(), y_train)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')


#%% Set the model to evaluation mode
model.eval()

# Make predictions on the test set
with torch.no_grad():
    predictions = model(X_test)

# Inverse scale the predicted stock prices
predicted_stock_price = scaler.inverse_transform(np.concatenate((predictions.numpy(), np.zeros((predictions.shape[0], 1))), axis=1))[:, 0]

# Inverse scale the actual test stock prices
real_stock_price = scaler.inverse_transform(np.concatenate((y_test.numpy().reshape(-1, 1), np.zeros((y_test.shape[0], 1))), axis=1))[:, 0]

plt.figure(figsize=(10, 6))
plt.plot(real_stock_price, label='Real Stock Price')
plt.plot(predicted_stock_price, label='Predicted Stock Price', linestyle='dashed')
plt.title('Seq2Seq LSTM Model: Real vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

#%% Plotting both vars against time

# fig, ax1 = plt.subplots(figsize=(10, 6))
#
# ax1.plot(df_joined_clean['Close'])
# ax2 = ax1.twinx()
# ax2.plot(df_joined_clean['label_numeric'])
# plt.show()
#
# plt.plot(df_joined_clean['Close'])
# plt.plot(df_joined_clean['label_numeric'])
# plt.show()

#%% Creating DF and First Order Differencing


# def differencing(data):
#     diff_data = []
#     for i in range(len(data)):
#         if i == 0:
#             diff_data.append(np.nan)
#         else:
#             diff_data.append((data[i]) - data[i-1])
#     return diff_data

# first order diff
# stock_df['1st_order_diff'] = differencing(stock_df['Close'])
# ACF_PACF_Plot(stock_df['1st_order_diff'].dropna(), 30, suptitle='ACF/PACF After 1st Order Diff ACF')
# Cal_rolling_mean_var(stock_df['1st_order_diff'].dropna(), '1st Order Differencing')
# ADF_Cal(stock_df['1st_order_diff'].dropna())
# kpss_test(stock_df['1st_order_diff'].dropna())

#%% splitting data
yt_stock, yf_stock = train_test_split(stock_df_clean['Close'].dropna(), shuffle=False, test_size=0.2)


#%% AR/ARX model --> USE THIS

steps_ahead = 197
out = f'AIC: {0:0.3f}, HQIC: {1:0.3f}, BIC: {2:0.3f}'


model = AutoReg(stock_df_clean['1st_order_diff'], lags=[1]).fit()
forecast = model.predict(start=len(yt_stock), end=len(yf_stock) + steps_ahead - 1)
print(out.format(model.aic, model.hqic, model.bic))

# res = AutoReg(stock_values, lags = [2]).fit()
# print(out.format(res.aic, res.hqic, res.bic))

model2 = AutoReg(endog=df_joined_clean['Close'], lags=[1],exog=df_joined_clean['label_numeric']).fit()

#%% Plotting data vs forecast

plt.plot(stock_df_clean['1st_order_diff'], label='Actual Data')
plt.plot(forecast, label='Forecast', linestyle='--')
plt.title('AR Model: Actual Data vs Forecast')
plt.xlabel('Date')
plt.xticks(rotation=30)
plt.ylabel('Values')
plt.legend()
plt.show()

# model performs better with no differencing vs with 1st order differencing (maybe data is stationary?)

#%% Plotting Trend, Seasonality, and Residuals

# STL Decomposition (stationary)
stl = STL(stock_values) #ACF peaks every 24 lags
res = stl.fit()
plt.figure(figsize=(10, 8))
fig = res.plot()
for ax in fig.get_axes():
    ax.tick_params(axis='x', labelsize=6.5) # Resize x-axis labels for all subplots
plt.tight_layout()
plt.show()

#%% Calculating Trend and Seasonality Strength

T = res.trend
S = res.seasonal
R = res.resid

def str_trend_seasonal(T, S, R):
    F = np.maximum(0, 1 - (np.var(np.array(R)) / np.var(np.array(T + R))))
    print(f'Trend strength is {100 * F: .3f}%')  # returns percentage
    FS = np.maximum(0, 1 - (np.var(np.array(R)) / np.var(np.array(S + R))))
    print(f'Seasonality strength is {100 * FS: .3f}%')  # returns percentage

str_trend_seasonal(T, S, R)

#%% ACF of stock data

def Cal_autocorrelation(data, max_lag, title):
    def autocorrelation(data, max_lag):
        n = len(data)
        mean = np.mean(data)
        numerator = sum((data[i] - mean) * (data[i - max_lag] - mean) for i in range(max_lag, n))
        denominator = sum((data[i] - mean) ** 2 for i in range(n))
        ry = numerator / denominator if denominator != 0 else 1.0
        return ry

    acf_values = [autocorrelation(data, lag) for lag in range(max_lag + 1)]

    a = acf_values
    b = a[::-1]
    c = b + a[1:]
    plt.figure()
    x_values = range(-max_lag, max_lag + 1)
    (markers, stemlines, baseline) = plt.stem(x_values, c, markerfmt='o')
    plt.setp(markers, color='red')
    m = 1.96 / np.sqrt(len(data))
    plt.axhspan(-m, m, alpha=0.2, color='blue')
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title(title)
    plt.show()

Cal_autocorrelation(stock_values, 100, 'JNJ Stock ACF')


#%% NOTES

# ADD MORE HEADLINES!!! (write iterative code for searching through more FT pages)
# code needs to be modular --> have one separate code for classes! --> modify github!
# READ ME file --> progress report (action items) --> separate readme file
# templates --> write up everytime I do something
# evalutate AR in the process (AIC + BIC)
# check AR results with reza
# exog input --> 3-time step of stock and data