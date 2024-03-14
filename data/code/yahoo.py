import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials

DJI_df = yf.download('DJI', 
                      start='2008-08-08', 
                      end='2016-01-01', 
                      progress=False,
)


# We only need the adjusted close price and the date
# Save the data to a CSV file
DJI_df = DJI_df[['Adj Close']]
# print(DJI_df.head())
DJI_df.to_csv('data/DJI_08-08-2008_01-01-2016_prices.csv')