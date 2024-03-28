import pandas as pd
import numpy as np
import sys

# Set print options to ensure all elements of an array are printed
np.set_printoptions(threshold=sys.maxsize)

# Load the data from the csv files
headlines_data = pd.read_csv('data/headlines_processed.csv')
dji_data = pd.read_csv('data/DJI_99to23_prices.csv')

# Convert 'Date' column to datetime
dji_data['Date'] = pd.to_datetime(dji_data['Date'])
headlines_data['Date'] = pd.to_datetime(headlines_data['Date'])

# Ensure data is sorted by date
dji_data = dji_data.sort_values(by='Date').reset_index(drop=True)
headlines_data = headlines_data.sort_values(by='Date').reset_index(drop=True)

# Filter the dji_data from 2018-01-02 to 2020-06-03
start_date = pd.to_datetime('2006-11-01')
end_date = pd.to_datetime('2023-12-31')
filtered_dji_data = dji_data[(dji_data['Date'] >= start_date) & (dji_data['Date'] <= end_date)]

print(filtered_dji_data.head())

# Initialize a list to hold the final structured data
final_data = []

# Filter the dji_data from 2018-01-02 to 2020-06-03 and explicitly create a copy
filtered_dji_data = dji_data[(dji_data['Date'] >= start_date) & (dji_data['Date'] <= end_date)].copy()

# Now, when you set a new column on this filtered DataFrame, it modifies the copy directly without warning
filtered_dji_data['Prev Adj Close'] = filtered_dji_data['Adj Close'].shift(1)

# # Loop through the filtered DJI data
# for index, row in filtered_dji_data.iterrows():
#     # Get the previous 30 days' prices
#     start_idx = max(index - 30, 0)
#     temp_df = filtered_dji_data.iloc[start_idx:index]
#     prices = temp_df['Adj Close'].tolist()
    
#     # Pad with avg of other days if there are not enough days
#     if len(prices) < 30:
#         pad_width = 30 - len(prices)
#         avg = np.mean(prices)
#         prices = np.pad(prices, (pad_width, 0), 'constant', constant_values=(avg, prices[0]))
    
#     # Fetch the corresponding day's headlines
#     daily_headlines = headlines_data[headlines_data['Date'] == row['Date']]['headline'].tolist()
    
#     # Choose how many headlines to consider
#     daily_headlines = daily_headlines[:25]

#     # print(daily_headlines)
#     # print(row['Date'])
    
#     y = 1 if (pd.notnull(row['Prev Adj Close']) and row['Adj Close'] > row['Prev Adj Close']) else 0
    
#     # Append to final_data
#     final_data.append({
#         'date': row['Date'],
#         'prices': prices,
#         'headlines': daily_headlines,
#         'y': y
#     })


# Loop through the filtered DJI data
for index, row in filtered_dji_data.iterrows():
    pre30, post7 = [], []
    # loop through the previous 30 days and the next 7 days and get the prices of each day
    if index >= 30:
        pre30 = filtered_dji_data.loc[index -30:index -1, 'Adj Close'].tolist()

    post7 = filtered_dji_data.loc[index + 1: index + 7, 'Adj Close'].tolist()
    
    # Fetch the corresponding day's headlines
    daily_headlines = headlines_data[headlines_data['Date'] == row['Date']]['headline'].tolist()
    
    # Choose how many headlines to consider
    daily_headlines = daily_headlines[:25]
    # divide the headlines into 5 parts, which will be 5 * 5 = 25
    # daily_headlines = [daily_headlines[i:i+5] for i in range(0, len(daily_headlines), 5)]
    

    # 5 parts share the same target
    # if len(daily_headlines) == 5:
    #     for i in range(5):

    y = 1 if (pd.notnull(row['Prev Adj Close']) and row['Adj Close'] > row['Prev Adj Close']) else 0
    # Append to final_data
    final_data.append({
        'date': row['Date'],
        'pre30': pre30,
        'post7': post7,
        'headlines': daily_headlines,
        'y': y
    })

    # print(daily_headlines)
    # print(row['Date'])


# Convert final_data to DataFrame for easier handling/manipulation
final_df = pd.DataFrame(final_data)

# Display the first few rows to verify the structure
# final_df.head()

# Save the final_df to a csv file
final_df.to_csv('data/final_dataset/dataset_without_embedding.csv', index=False)
