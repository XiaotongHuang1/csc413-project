import requests
import json
from threading import Thread
import csv
import math


def fetch_data(start_id, end_id, page_size, total_items, headlines, date_publisheds):
    url = "https://webql-redesign.cnbcfm.com/graphql"
    offset = 0
    _id = start_id

    variables_template = '{{"id":{},"offset":{},"pageSize":{},"nonFilter":true,"includeNative":false,"include":[]}}'
    extensions = '%7B%22persistedQuery%22%3A%7B%22version%22%3A1%2C%22sha256Hash%22%3A%2243ed5bcff58371b2637d1f860e593e2b56295195169a5e46209ba0abb85288b7%22%7D%7D'

    while _id <= end_id and len(headlines) < total_items:
        variables = variables_template.format(_id, offset, page_size)
        encoded_variables = variables.replace(' ', '%20')
        full_url = f"{url}?operationName=getAssetList&variables={encoded_variables}&extensions={extensions}"

        response = requests.get(full_url)
        response.raise_for_status()

        data = response.json()
        assets = []
        try:
            assets = data['data']['assetList']['assets']
        except:
            print('No data found. Exiting for id:', _id)

        if assets != []:
            print(f"Data found for id: {_id}")
            for asset in assets:
                headlines.append(asset['headline'])
                date_publisheds.append(asset['datePublished'])

            offset += page_size
            print(f"Retrieved {len(headlines)} headlines from id: {_id}")
        else:
            print(f"No data found. Moving to next id: {_id + 1}")
            _id += 1
            offset = 0

def main():
    page_size = 24
    total_items = 999999999
    num_threads = 100
    start_id = 10000000
    id_range = 10

    headlines = []
    date_publisheds = []

    threads = []
    for i in range(num_threads):
        thread_start_id = start_id + i * id_range
        thread_end_id = thread_start_id + id_range - 1
        thread = Thread(target=fetch_data, args=(thread_start_id, thread_end_id, page_size, total_items, headlines, date_publisheds))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    headlines = headlines[:total_items]
    date_publisheds = date_publisheds[:total_items]

    print(f"TOTAL: Retrieved {len(headlines)} headlines.")

    with open('cnbc_headlines.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['headline', 'datePublished'])
        writer.writerows(zip(headlines, date_publisheds))

if __name__ == "__main__":
    main()

