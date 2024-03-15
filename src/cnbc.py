import requests
import json
from datetime import datetime, timedelta
from urllib.parse import unquote
import csv
import threading

"""
This script demonstrates how to scrape headlines from CNBC using their GraphQL API.
Usage:
Get headlines for a range of dates:
between 2000-01-01 and 2024-03-15 and save them to a CSV file
"""

def get_headlines_for_date(date):
    url = "https://webql-redesign.cnbcfm.com/graphql"
    variables = json.loads(unquote('%7B%22id%22%3A%2210000664%22%2C%22offset%22%3A0%2C%22pageSize%22%3A100%2C%22nonFilter%22%3Atrue%2C%22includeNative%22%3Afalse%2C%22include%22%3A%5B%5D%2C%22filter%22%3A%7B%22and%22%3A%5B%7B%22range%22%3A%7B%22datePublished%22%3A%7B%22gte%22%3A%22' + date + 'T00%3A00%3A00%2B0000%22%2C%22lte%22%3A%22' + date + 'T23%3A59%3A59%2B0000%22%7D%7D%7D%5D%7D%7D'))
    params = {
        "operationName": "getAssetList",
        "variables": json.dumps(variables),
        "extensions": '{"persistedQuery":{"version":1,"sha256Hash":"43ed5bcff58371b2637d1f860e593e2b56295195169a5e46209ba0abb85288b7"}}'
    }
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/json",
        "Origin": "https://www.cnbc.com",
        "Referer": "https://www.cnbc.com/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    if 'errors' in data:
        print(f'Error fetching headlines for {date}: {data["errors"]}')
        return []
    headlines = [asset['headline'] for asset in data['data']['assetList']['assets']]
    return headlines

def get_headlines_for_year(year):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    if year == 2024:
        end_date = datetime(year, 3, 15)
    current_date = start_date
    all_headlines = {}

    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f'Fetching headlines for {date_str}')
        headlines = get_headlines_for_date(date_str)
        all_headlines[date_str] = headlines
        current_date += timedelta(days=1)

    return all_headlines


def collect_headlines(start_date, end_date, output_file):
    current_date = start_date
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['headline', 'date'])  # Write the header

        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            print(f'Fetching headlines for {date_str}')
            headlines = get_headlines_for_date(date_str)
            writer.writerows(headlines)
            current_date += timedelta(days=1)


def write_headlines_to_csv(year, headlines):
    with open(f'headlines_{year}.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['date', 'headline'])  # Write the header
        for date, daily_headlines in headlines.items():
            for headline in daily_headlines:
                writer.writerow([date, headline])
    print(f'Wrote headlines for {year} to headlines_{year}.csv')


def fetch_and_write_headlines(year):
    headlines = get_headlines_for_year(year)
    write_headlines_to_csv(year, headlines)


if __name__ == "__main__":
    # threads = []
    # for year in range(2000, 2024):
    #     thread = threading.Thread(target=fetch_and_write_headlines, args=(year,))
    #     threads.append(thread)
    #     thread.start()

    # for thread in threads:
    #     thread.join()

    # combine all the csv files into one
    with open('financial_headlines.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['date', 'headline'])  # Write the header
        for year in range(2000, 2024):
            with open(f'headlines_{year}.csv', 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # skip the header
                writer.writerows(reader)