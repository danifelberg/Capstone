import requests
import pandas as pd
from bs4 import BeautifulSoup
import time


# -*- coding: utf-8 -*-
"""
Author: Daniel Felberg
Date: 2024-09-23
Version: 1.0
"""

def scrape_ft_headlines(company, num_pages, delay=2):
    all_headlines = []
    all_dates = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive'
    }

    for page_num in range(1, num_pages + 1):
        # Adjust the URL to the current page number
        url = f"https://www.ft.com/search?q={company}&page={page_num}&sort=relevance&isFirstView=false"
        response = requests.get(url, headers=headers)

        # Check if the response was successful
        print(f"Fetching page {page_num}: Status code {response.status_code}")

        if response.status_code != 200:
            print(f"Error fetching page {page_num}.")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')

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

        time.sleep(delay)

    # Find the first and last dates
    if all_dates:
        first_date = min(filter(None, all_dates))  # Filter out None values
        last_date = max(filter(None, all_dates))
    else:
        first_date, last_date = None, None

    dates = pd.to_datetime(all_dates)
    df = pd.DataFrame({'Headline': all_headlines},index=dates)

    df_index = df.reset_index()
    df_noNA = df_index.dropna(subset=['index'])
    df_noNA_index = df_noNA.set_index('index')
    print(f'Headlines scraped: {len(df_noNA_index)}')

    return df_noNA_index, first_date, last_date