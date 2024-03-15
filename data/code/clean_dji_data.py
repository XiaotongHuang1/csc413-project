import pandas as pd

# load csv file
headlines_data = pd.read_csv('data/dow-jones-industrial-average-last-10-years.csv')

# get the prices from 2011-01-02 to 2020-06-03
start_date = pd.to_datetime('2011-03-14')
end_date = pd.to_datetime('2020-06-03')
# and having 30 days before the start date
start_date = start_date - pd.DateOffset(days=30)
# convert the 'Date' column to datetime
headlines_data['Date'] = pd.to_datetime(headlines_data['Date'])
filtered_dji_data = headlines_data[(headlines_data['Date'] >= start_date) & (headlines_data['Date'] <= end_date)]

# save the data to a csv file
filtered_dji_data.to_csv('data/DJI_11to20_prices.csv', index=False)