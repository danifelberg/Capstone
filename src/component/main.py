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
from utils_models import ARIMA_stock, LSTM_no_headlines, SARIMAX_stock, SARIMA_split, LSTM_headlines
from utils_stationarity import evaluate_models, rev_diff
from src.component.utils_EDA import ARIMA_viz, LSTM_viz, SARIMAX_viz, LSTM_headlines_viz
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
label_map = {"neutral": 0, "negative": -1, "positive": 1}
headlines['numeric_labels'] = headlines['Label'].map(label_map)
print(headlines.head(5))

# used for padding LSTM w/ headlines
most_frequent_date = headlines.index.value_counts().idxmax()
most_frequent_date_count = headlines.index.value_counts().max()

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
rmse_LSTM,yf_LSTM,yf_unscaled_LSTM,predictions_LSTM = LSTM_no_headlines(stock_values)
# print(f'RMSE for LSTM: {round(rmse_LSTM,4)}')

LSTM_train_size = int(len(stock_values) * 0.8)
LSTM_viz(stock_values,LSTM_train_size, yf_unscaled_LSTM,predictions_LSTM,ticker_symbol)

#%% LSTM (headlines)
rmse_LSTM_headlines, y_train_headlines,y_test_headlines,predictions_headlines = LSTM_headlines(stock_values,headlines,most_frequent_date_count)
print(f'RMSE for LSTM ({company}, no headlines): {round(rmse_LSTM,4)}')
print(f'RMSE for LSTM ({company}, headlines): {round(rmse_LSTM_headlines,4)}')

LSTM_headlines_viz(y_train_headlines,y_test_headlines,predictions_headlines,ticker_symbol)

# better performance: Nvidia, Apple, Microsoft, Amazon, Google (Alphabet),
# Aramco, Facebook/META, Berkshire Hathaway, TSMC, Tesla

#%% LSTM (concat)


#%% SARIMAX

# SARIMA_yt,SARIMA_yf,exog_yt,exog_yf = SARIMA_split(combined_df)
# SARIMA_res, SARIMA_predictions = SARIMAX_stock(SARIMA_yt, SARIMA_yf, (2,1,2), (0,0,0,0), exog_yt, exog_yf)
#
# SARIMAX_viz(SARIMA_yt,SARIMA_yf,SARIMA_res,SARIMA_predictions)

#%% NOTES

# ADD MORE HEADLINES!!! (develop code for other sources)
# check AR results with reza
# exog input --> automate pipeline for ARX after AR is finalized

