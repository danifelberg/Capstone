#%%
import sys
import warnings
sys.path.append(r'C:\Users\danif\Capstone_Group_2\src\component')
#%%
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from src.component.class_scrapers import scrape_ft_headlines
from src.component.utils_stationarity import Cal_autocorrelation, ACF_PACF_Plot, grid_search
from src.component.utils_stationarity import Cal_rolling_mean_var, differencing
from src.component.class_stocks import fetch_stock_data, plot_stock_data
from utils_models import ARIMA_stock, LSTM, SARIMAX_stock, SARIMA_split, LSTM_headlines, LSTM_headlines_new
from utils_stationarity import evaluate_models, rev_diff
from src.component.utils_EDA import ARIMA_viz, LSTM_viz, SARIMAX_viz
from datetime import datetime, timedelta
from src.component.sentiment_analysis import sentiment_analysis,analyze_sentiment
from statsmodels.tsa.api import SARIMAX

#%% define parameters
company = str(input("Company name (e.g. J&J, Ford, Microsoft, etc.):")) # company name to be scraped for news headlines
ticker_symbol = str(input("Corresponding company ticker symbol:")) # corresponding company name to be used (from yahoo finance)

#%% scraping Financial Times ONLY
headlines, start_date, end_date = scrape_ft_headlines(company,
                    int(input("Number of pages in search results:")))  # try not to exceed 10

#%% sentiment analysis
headlines['Label'],headlines['Score'] = sentiment_analysis(headlines['Headline'])
label_map = {"positive": 1, "neutral": 2, "negative": 3}
headlines['numeric_labels'] = headlines['Label'].map(label_map)
print(headlines.head(5))

#%% fetching stock data and EDA
# start_date = (pd.to_datetime(datetime.now()) - timedelta(days=180)).date() # TEMPORARY
# end_date = pd.to_datetime(datetime.now()).date() # TEMPORARY
stock_values = fetch_stock_data(ticker_symbol, start_date, end_date, interval='1d')
plot_stock_data(stock_values,company)

#%% stationarity
ACF_PACF_Plot(np.array(stock_values),50,f'ACF/PACF of {company} stock')
Cal_rolling_mean_var(stock_values,f'{company} stock',len(stock_values))

#%% first order diff
diff_stock = stock_values.diff().dropna()
Cal_autocorrelation(diff_stock,100,f'{company} Stock ACF (1st Order Diff)')

#%% checking stationarity again
ACF_PACF_Plot(np.array(diff_stock),50,f'ACF/PACF of {company} stock (1st Ord Diff)')
Cal_rolling_mean_var(diff_stock,f'{company} Stock (1st Ord Diff)',len(stock_values))

#%% Grid search for ARIMA (stationary data)
p_values = [0, 1, 2, 4, 6]
d_values = range(0, 3)
q_values = range(0, 3)

warnings.filterwarnings("ignore")
evaluate_models(diff_stock.values, p_values, d_values, q_values)

#%% ARIMA
warnings.filterwarnings("ignore")
y = diff_stock
order = (2,1,2)
yt_stock,yf_stock,predictions,rmse = ARIMA_stock(y,order)
print(f'RMSE for order ARIMA{order}: {rmse}')

# plotting ARIMA(2,1,2) model
ARIMA_viz(stock_values,yt_stock,yf_stock,predictions,order)

#%% LSTM (no headlines)
warnings.filterwarnings("ignore")
# raw data
rmse_LSTM,yf_LSTM,yf_unscaled_LSTM,predictions_LSTM = LSTM(stock_values)
print(f'RMSE for LSTM: {round(rmse_LSTM,4)}')

LSTM_train_size = int(len(stock_values) * 0.8)
LSTM_viz(stock_values,LSTM_train_size, yf_unscaled_LSTM,predictions_LSTM)

#%% LSTM (headlines)
stock_df = stock_values.to_frame(name='Close')
stock_df.index.name = 'Date'
headlines.index.name = 'Date'

headlines_grouped = headlines.groupby(headlines.index)['numeric_labels'].apply(list).reset_index()
headlines_grouped.set_index('Date', inplace=True)
combined_df = headlines_grouped.join(stock_df, how='outer')

combined_noNA = combined_df.dropna(subset=['Close'])
combined_noNA['numeric_labels'] = combined_noNA['numeric_labels'].apply(lambda x: [0] if isinstance(x, float) and pd.isna(x) else x)

from tensorflow.keras.preprocessing.sequence import pad_sequences
combined_noNA['numeric_labels_padded'] = list(
    pad_sequences(combined_noNA['numeric_labels'], maxlen=5, padding='post', value=0)
)

time_steps = 1
X_close = []
X_numeric_labels = []
y_close = []

for i in range(time_steps, len(combined_noNA)):
    X_close.append(combined_noNA['Close'].iloc[i-time_steps:i].values)
    X_numeric_labels.append(np.array(combined_noNA['numeric_labels_padded'].iloc[i-time_steps:i].tolist()))
    y_close.append(combined_noNA['Close'].iloc[i])

X_close = np.expand_dims(X_close, axis=2)  # Add a third dimension to X_close
X_numeric_labels = np.array(X_numeric_labels)
X_numeric_labels = np.mean(X_numeric_labels, axis=2, keepdims=True)  # combining using mean
y_close = np.array(y_close)  # Shape: (samples,)

X_combined = np.concatenate([X_close, X_numeric_labels], axis=2)

split_index = int(len(X_combined) * 0.8)
X_train, X_test = X_combined[:split_index], X_combined[split_index:]
y_train, y_test = y_close[:split_index], y_close[split_index:]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
X_train_scaled = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test_scaled = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.reshape(-1, 1))

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train_scaled, y_train_scaled, epochs=20, batch_size=32, validation_data=(X_test_scaled, y_test_scaled))
predictions_scaled = model.predict(X_test_scaled)
predictions = scaler.inverse_transform(predictions_scaled)

import matplotlib.pyplot as plt

train_size = len(y_train)
test_size = len(y_test)
train_data = np.concatenate([y_train, np.full_like(y_test, np.nan)])  # Combine y_train and NaN for the test portion
test_data = np.concatenate([np.full_like(y_train, np.nan), y_test])  # Combine NaN for the train portion and y_test

plt.figure(figsize=(14,6))
plt.plot(train_data, label='Training Data (Actual)', color='blue')
plt.plot(test_data, label='Test Data (Actual)', color='green')
plt.plot(np.arange(train_size, train_size + test_size), predictions, label='Predicted Data', color='red')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# rmse_LSTM_headlines,actual_LSTM_headlines,predictions_LSTM_headlines = LSTM_headlines(combined_df)
# print(f'RMSE for LSTM (headlines): {round(rmse_LSTM_headlines,4)}')
# LSTM_viz(stock_values,LSTM_train_size, actual_LSTM_headlines,predictions_LSTM_headlines)

predictions_unscaled,y_test_unscaled = LSTM_headlines_new(combined_noNA)

#%% SARIMAX

SARIMA_yt,SARIMA_yf,exog_yt,exog_yf = SARIMA_split(combined_df)
SARIMA_res, SARIMA_predictions = SARIMAX_stock(SARIMA_yt, SARIMA_yf, (2,1,2), (0,0,0,0), exog_yt, exog_yf)

SARIMAX_viz(SARIMA_yt,SARIMA_yf,SARIMA_res,SARIMA_predictions)

#%% NOTES

# ADD MORE HEADLINES!!! (develop code for other sources)
# check AR results with reza
# exog input --> automate pipeline for ARX after AR is finalized

