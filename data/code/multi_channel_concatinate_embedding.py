import pandas as pd
import numpy as np

def convert_embedding_string_to_array(embedding_str):
    # Split the string into a list of strings, each representing a float
    embedding_list_str = embedding_str.strip("[]").split()
    # Convert the list of strings to a list of floats
    embedding_list_float = [float(num) for num in embedding_list_str]
    # Convert the list of floats to a NumPy array
    return np.array(embedding_list_float)



# Load the data from the csv files
headline_embeddings_data = pd.read_csv('data/final_dataset/sorted_embedding_headlines_processed.csv')
dji_data = pd.read_csv('data/DJI_99to23_prices.csv')

dji_data['Date'] = pd.to_datetime(dji_data['Date'])
headline_embeddings_data['Date'] = pd.to_datetime(headline_embeddings_data['Date'])

# Convert the embeddings from string representation back to arrays
headline_embeddings_data['embedding'] = headline_embeddings_data['embedding'].apply(convert_embedding_string_to_array)

# Ensure data is sorted by date
dji_data = dji_data.sort_values(by='Date').reset_index(drop=True)
headlines_data = headline_embeddings_data.sort_values(by='Date').reset_index(drop=True)

# Filter the dji_data from 2018-01-02 to 2020-06-03
# start_date = pd.to_datetime('2007-01-01')
# end_date = pd.to_datetime('2023-12-31')
# filtered_dji_data = dji_data[(dji_data['Date'] >= start_date) & (dji_data['Date'] <= end_date)]


# filtered_dji_data = dji_data[(dji_data['Date'] >= start_date) & (dji_data['Date'] <= end_date)].copy()
filtered_dji_data = dji_data[(dji_data['Date'] >= pd.to_datetime('2007-01-01')) & (dji_data['Date'] <= pd.to_datetime('2023-12-31'))].copy()
print(filtered_dji_data.tail())

# Now, when you set a new column on this filtered DataFrame, it modifies the copy directly without warning
filtered_dji_data['Prev Adj Close'] = filtered_dji_data['Adj Close'].shift(1)

# Initialize a list to hold the final structured data
final_data_with_embeddings = []

print(filtered_dji_data.sample(10))

for index, row in filtered_dji_data.iterrows():
    # Extract the current date's embeddings
    current_date_embeddings = headline_embeddings_data[headline_embeddings_data['Date'] == row['Date']]['embedding'].values

    # Handle embedding logic (as before)
    if len(current_date_embeddings) > 0:
        embeddings = np.array([embedding for embedding in current_date_embeddings[:25]])
        if len(embeddings) < 25:
            padding = np.mean(embeddings, axis=0)
            additional_required = 25 - len(embeddings)
            padding = np.tile(padding, (additional_required, 1))
            embeddings = np.vstack((embeddings, padding))
    
    start_idx = max(index - 30, 0)
    temp_df = filtered_dji_data.iloc[start_idx:index]

    # Calculate 'prices' based on available data in 'temp_df'
    prices = temp_df['Adj Close'].tolist()

    if len(prices) < 30:
        pad_width = 30 - len(prices)
        # Use the available 'prices' average or a default if none are available
        avg = np.mean(prices) if len(prices) > 0 else row['Adj Close']  # Fallback to current row's price
        prices = np.pad(prices, (pad_width, 0), 'constant', constant_values=(avg, avg))

    y = 1 if (pd.notnull(row['Prev Adj Close']) and row['Adj Close'] > row['Prev Adj Close']) else 0

    final_data_with_embeddings.append({
        'date': row['Date'],
        'prices': prices,
        'headlines': embeddings.flatten(),
        'y': y
    })



# Convert final_data_with_embeddings to DataFrame for easier handling/manipulation
final_df_with_embeddings = pd.DataFrame(final_data_with_embeddings)

# Display the structure of the final DataFrame
print(final_df_with_embeddings.head())

# Display the first few rows to verify the structure
# final_df.head()

# Save the final_df to a csv file
final_df_with_embeddings.to_csv('data/final_dataset/dataset1.csv', index=False)
