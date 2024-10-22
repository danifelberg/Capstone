import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta


# -*- coding: utf-8 -*-
"""
Author: Daniel Felberg
Date: 2024-09-23
Version: 1.0
"""


def scrape_ft_headlines(company, num_pages, delay=5):
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

    print(f'Headlines scraped: {len(all_headlines)}')
    return all_headlines, all_dates, first_date, last_date


def get_economist_headlines(query, max_pages=5):
    """
    Scrapes article headlines and dates from The Economist search results.

    Parameters:
    - query: The search term to look for.
    - max_pages: The number of result pages to iterate through.

    Returns:
    - A list of tuples containing the article headline and the corresponding date.
    """
    base_url = f"https://www.economist.com/search?q={query}&page={num_pages}"
    headlines = []

    for page_num in range(1, max_pages + 1):
        url = base_url.format(query=query, page_num=page_num)
        print(f"Scraping page {page_num}: {url}")

        # Send the request to the search results page
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Failed to retrieve page {page_num}. Status code: {response.status_code}")
            break

        # Parse the page content
        soup = BeautifulSoup(response.content, "html.parser")

        # Find all articles in the search results
        articles = soup.find_all("article")

        if not articles:
            print(f"No articles found on page {page_num}.")
            break

        # Extract headlines and dates
        for article in articles:
            headline = article.find("a", {"class": "headline-link"}).get_text(strip=True)
            date = article.find("time").get_text(strip=True)

            headlines.append((headline, date))

        # Sleep for a bit to be polite to the server
        time.sleep(1)

    return headlines