# -*- coding: utf-8 -*-
"""
Author: Daniel Felberg
Date: 2024-10-13
Version: 1.0
"""

import yfinance as yf
import matplotlib.pyplot as plt


def fetch_stock_data(company, min_date, max_date, interval='1d'):
    """
    Fetches the stock data for the specified company and date range.

    Parameters:
    - company: The ticker symbol for the company stock (e.g., 'JNJ').
    - min_date: The start date for fetching stock data (format: 'YYYY-MM-DD').
    - max_date: The end date for fetching stock data (format: 'YYYY-MM-DD').
    - interval: The interval for the stock data (default: '1d' for daily data).

    Returns:
    - A pandas Series of the closing stock values.
    """
    stock_data = yf.download(company, start=min_date, end=max_date, interval=interval)
    stock_values = stock_data['Close']
    return stock_values




def plot_stock_data(stock_values,company):
    """
    Plots the closing stock values over time.

    Parameters:
    - stock_values: A pandas Series of the stock's closing values.
    """
    if stock_values is None or stock_values.empty:
        print("No stock data available.")
        return

    plt.plot(stock_values)
    plt.xlabel("Date")
    plt.ylabel("Closing Stock Value")
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{company} Daily Closing Stock Values')
    plt.tight_layout()
    plt.show()


def check_equal_sampling(stock_values):
    """
    Checks whether the stock data is equally sampled in time.

    Parameters:
    - stock_values: A pandas Series of the stock's closing values.

    Returns:
    - A pandas Series with the time differences between consecutive observations.
    """
    if stock_values is None or stock_values.empty:
        print("No stock data available.")
        return

    # Check time differences between consecutive observations
    time_diffs = stock_values.index.to_series().diff().dropna()

    if time_diffs.nunique() == 1:
        print("The data is equally sampled.")
    else:
        print("The data is not equally sampled.")
        print(time_diffs.value_counts())

    return time_diffs



# class StockAnalyzer:
#     def __init__(self, company, min_date, max_date, interval='1d'):
#         """
#         Initializes the StockAnalyzer class.
#
#         Parameters:
#         - company: The ticker symbol for the company stock (e.g., 'JNJ').
#         - min_date: The start date for fetching stock data (format: 'YYYY-MM-DD').
#         - max_date: The end date for fetching stock data (format: 'YYYY-MM-DD').
#         - interval: The interval for the stock data (default: '1d' for daily data).
#         """
#         self.company = company
#         self.min_date = min_date
#         self.max_date = max_date
#         self.interval = interval
#         self.stock_data = None
#         self.stock_values = None
#
#     def fetch_stock_data(self):
#         """
#         Fetches the stock data for the specified company and date range.
#         """
#         self.stock_data = yf.download(self.company, start=self.min_date, end=self.max_date, interval=self.interval)
#         self.stock_values = self.stock_data['Close']
#         return self.stock_values
#
#     def plot_stock_data(self):
#         """
#         Plots the closing stock values over time.
#         """
#         if self.stock_values is None:
#             print("No stock data available. Please fetch the stock data first.")
#             return
#
#         plt.plot(self.stock_values)
#         plt.xlabel("Date")
#         plt.ylabel("Closing Stock Value")
#         plt.tight_layout()
#         plt.show()
#
#     def check_equal_sampling(self):
#         """
#         Checks whether the stock data is equally sampled in time.
#         """
#         if self.stock_values is None:
#             print("No stock data available. Please fetch the stock data first.")
#             return
#
#         # Check time differences between consecutive observations
#         time_diffs = self.stock_values.index.to_series().diff().dropna()
#
#         if time_diffs.nunique() == 1:
#             print("The data is equally sampled.")
#         else:
#             print("The data is not equally sampled.")
#             print(time_diffs.value_counts())
#
#         return time_diffs

# class Person:
#     def __init__(self, name, age) -> object:
#         """
#
#         :rtype: object
#         :param name:
#         :param age:
#         """
#         self.name = name
#         self.age = age
#
#     def __str__(self):
#         """
#
#         :rtype: object
#         """
#         return f"{self.name}({self.age})"
