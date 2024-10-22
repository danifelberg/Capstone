#%%
import sys
import warnings
sys.path.append(r'C:\Users\danif\Capstone_Group_2\src\component')
#%%
import pandas as pd
import numpy as np
from src.component.class_scrapers import scrape_ft_headlines
from src.component.utils_stationarity import Cal_autocorrelation, ACF_PACF_Plot
from src.component.utils_stationarity import Cal_rolling_mean_var, differencing
from src.component.class_stocks import fetch_stock_data, plot_stock_data
from utils_models import ARIMA_stock
from utils_stationarity import evaluate_models
from utils_EDA import ARIMA_viz
from datetime import datetime, timedelta
from src.component.sentiment_analysis import sentiment_analysis

#%% define parameters
company = str(input("Company name (e.g. J&J, Ford, Microsoft, etc.):")) # company name to be scraped for news headlines
ticker_symbol = str(input("Corresponding company ticker symbol:")) # corresponding company name to be used (from yahoo finance)

#%% scraping Financial Times ONLY
headlines, dates, start_date, end_date = scrape_ft_headlines(company,
                    int(input("Number of pages in search results:")))  # try not to exceed 10

#%% fetching stock data and EDA
# start_date = (pd.to_datetime(datetime.now()) - timedelta(days=180)).date() # TEMPORARY
# end_date = pd.to_datetime(datetime.now()).date() # TEMPORARY
stock_values = fetch_stock_data(ticker_symbol, start_date, end_date, interval='1d')
plot_stock_data(stock_values,company)

#%% stationarity
ACF_PACF_Plot(np.array(stock_values),50,f'ACF/PACF of {company} stock')
Cal_rolling_mean_var(stock_values,f'{company} stock',len(stock_values))

#%% first order diff
diff_stock = differencing(stock_values)
diff_stock_clean = [x for x in diff_stock if not pd.isna(x)]
Cal_autocorrelation(diff_stock_clean,100,f'{company} Stock ACF')

#%% checking stationarity again
diff_stock_clean_df = pd.DataFrame(diff_stock_clean)
ACF_PACF_Plot(np.array(diff_stock_clean_df),50,f'ACF/PACF of {company} stock (1st Ord Diff)')
Cal_rolling_mean_var(diff_stock_clean_df,f'{company} Stock (1st Ord Diff)',len(stock_values))

#%% Grid search for ARIMA (stationary data)
p_values = [0, 1, 2, 4, 6]
d_values = range(0, 3)
q_values = range(0, 3)

warnings.filterwarnings("ignore")
evaluate_models(diff_stock_clean_df.values, p_values, d_values, q_values)

# Best ARIMA(2, 1, 2) RMSE=7.943, TSLA (stationary, 1st order diff) --> NOT ACTUALLY THE BEST

#%% modelling
warnings.filterwarnings("ignore")
y = stock_values
order = (1,0,0)
yt_stock,yf_stock,predictions,rmse = ARIMA_stock(y,order)

ARIMA_viz(stock_values,yt_stock,yf_stock,predictions,order)

#%% NOTES

# ADD MORE HEADLINES!!! (develop code for other sources)
# check AR results with reza
# exog input --> automate pipeline for ARX after AR is finalized