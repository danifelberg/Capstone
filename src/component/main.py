#%%
import sys
import warnings
sys.path.append(r'your_path_here')
#%%
warnings.filterwarnings("ignore")
from src.component.class_scrapers import scrape_ft_headlines
from src.component.class_stocks import fetch_stock_data, plot_stock_data
from utils_models import LSTM_no_headlines, LSTM_headlines
from src.component.utils_EDA import LSTM_viz, LSTM_headlines_viz
from src.component.sentiment_analysis import sentiment_analysis

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
stock_values = fetch_stock_data(ticker_symbol, start_date, end_date, interval='1d')
plot_stock_data(stock_values,company)

#%% LSTM (no headlines)
warnings.filterwarnings("ignore")

rmse_LSTM,yf_LSTM,yf_unscaled_LSTM,predictions_LSTM = LSTM_no_headlines(stock_values)

LSTM_train_size = int(len(stock_values) * 0.8)
LSTM_viz(stock_values,LSTM_train_size, yf_unscaled_LSTM,predictions_LSTM,ticker_symbol)

#%% LSTM (headlines)
rmse_LSTM_headlines, y_train_headlines,y_test_headlines,predictions_headlines = LSTM_headlines(stock_values,headlines,most_frequent_date_count)
print(f'RMSE for LSTM ({company}, no headlines): {round(rmse_LSTM,4)}')
print(f'RMSE for LSTM ({company}, headlines): {round(rmse_LSTM_headlines,4)}')

LSTM_headlines_viz(y_train_headlines,y_test_headlines,predictions_headlines,ticker_symbol)