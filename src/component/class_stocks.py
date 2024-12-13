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