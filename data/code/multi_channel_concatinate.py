import pandas as pd
import numpy as np

# Load the data from the csv files
headlines_data = pd.read_csv('data/clean_train_max25.csv')
dji_data = pd.read_csv('data/DJI_18to20_prices.csv')

# Convert 'Date' column to datetime
dji_data['Date'] = pd.to_datetime(dji_data['Date'])
headlines_data['Date'] = pd.to_datetime(headlines_data['Date'])

# Ensure data is sorted by date
dji_data = dji_data.sort_values(by='Date').reset_index(drop=True)
headlines_data = headlines_data.sort_values(by='Date').reset_index(drop=True)

# Filter the dji_data from 2018-01-02 to 2020-06-03
start_date = pd.to_datetime('2018-01-02')
end_date = pd.to_datetime('2020-06-03')
filtered_dji_data = dji_data[(dji_data['Date'] >= start_date) & (dji_data['Date'] <= end_date)]

print(filtered_dji_data.head())

# Initialize a list to hold the final structured data
final_data = []

# Filter the dji_data from 2018-01-02 to 2020-06-03 and explicitly create a copy
filtered_dji_data = dji_data[(dji_data['Date'] >= start_date) & (dji_data['Date'] <= end_date)].copy()

# Now, when you set a new column on this filtered DataFrame, it modifies the copy directly without warning
filtered_dji_data['Prev Adj Close'] = filtered_dji_data['Adj Close'].shift(1)

# Loop through the filtered DJI data
for index, row in filtered_dji_data.iterrows():
    # Get the previous 30 days' prices
    start_idx = max(index - 30, 0)
    temp_df = filtered_dji_data.iloc[start_idx:index]
    prices = temp_df['Adj Close'].tolist()
    
    # Pad with avg of other days if there are not enough days
    if len(prices) < 30:
        pad_width = 30 - len(prices)
        avg = np.mean(prices)
        prices = np.pad(prices, (pad_width, 0), 'constant', constant_values=(avg, prices[0]))
    
    # Fetch the corresponding day's headlines
    daily_headlines = headlines_data[headlines_data['Date'] == row['Date']]['headline'].tolist()
    
    # Ensure only the top 25 headlines are considered
    daily_headlines = daily_headlines[:5]

    # print(daily_headlines)
    # print(row['Date'])
    
    y = 1 if (pd.notnull(row['Prev Adj Close']) and row['Adj Close'] > row['Prev Adj Close']) else 0
    
    # Append to final_data
    final_data.append({
        'date': row['Date'],
        'prices': prices,
        'headlines': daily_headlines,
        'y': y
    })


# Convert final_data to DataFrame for easier handling/manipulation
final_df = pd.DataFrame(final_data)

# Display the first few rows to verify the structure
# final_df.head()

# Save the final_df to a csv file
final_df.to_csv('data/final_data_top5.csv', index=False)
