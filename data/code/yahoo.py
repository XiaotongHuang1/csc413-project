import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials

DJI_df = yf.download('DJI', 
                      start='1999-12-01', 
                      end='2011-03-14', 
                      progress=False,
)


# We only need the adjusted close price and the date
# Save the data to a CSV file
DJI_df = DJI_df[['Adj Close']]
# print(DJI_df.head())
DJI_df.to_csv('./data/DJI_99to11_prices.csv')