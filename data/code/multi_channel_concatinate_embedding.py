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
headline_embeddings_data = pd.read_csv('data\dataset_tmp.csv')
dji_data = pd.read_csv('data\DJI_18to20_prices.csv')

dji_data['Date'] = pd.to_datetime(dji_data['Date'])
headline_embeddings_data['Date'] = pd.to_datetime(headline_embeddings_data['Date'])

# Convert the embeddings from string representation back to arrays
headline_embeddings_data['embedding'] = headline_embeddings_data['embedding'].apply(convert_embedding_string_to_array)

# Ensure data is sorted by date
dji_data = dji_data.sort_values(by='Date').reset_index(drop=True)
headlines_data = headline_embeddings_data.sort_values(by='Date').reset_index(drop=True)

# Filter the dji_data from 2018-01-02 to 2020-06-03
start_date = pd.to_datetime('2018-01-02')
end_date = pd.to_datetime('2020-06-03')
filtered_dji_data = dji_data[(dji_data['Date'] >= start_date) & (dji_data['Date'] <= end_date)]

# Filter the dji_data from 2018-01-02 to 2020-06-03 and explicitly create a copy
filtered_dji_data = dji_data[(dji_data['Date'] >= start_date) & (dji_data['Date'] <= end_date)].copy()
# Now, when you set a new column on this filtered DataFrame, it modifies the copy directly without warning
filtered_dji_data['Prev Adj Close'] = filtered_dji_data['Adj Close'].shift(1)

# Initialize a list to hold the final structured data
final_data_with_embeddings = []

for index, row in filtered_dji_data.iterrows():
    # Fetch the corresponding day's headline embeddings
    current_date_embeddings = headline_embeddings_data[headline_embeddings_data['Date'] == row['Date']]['embedding'].values
    
    # Check if there are embeddings for the current date
    if len(current_date_embeddings) > 0:
        # Since each day may have multiple headlines and therefore multiple embeddings,
        # we need to handle potentially having more than 25 embeddings or fewer.
        embeddings = np.array([embedding for embedding in current_date_embeddings[:25]])

        # If there are fewer than 25 embeddings, pad with average embeddings    
        if len(embeddings) < 25:
            padding = np.mean(embeddings, axis=0)
            embeddings = np.vstack((embeddings, padding))
        else:
            # If there are more than 25, just use the first 25
            embeddings = embeddings[:25]

        # Flatten the embeddings to create a single feature vector
        embeddings_flattened = embeddings.flatten()

        # Get the previous 30 days' prices
        start_idx = max(index - 30, 0)
        temp_df = filtered_dji_data.iloc[start_idx:index]
        prices = temp_df['Adj Close'].tolist()
        
        # Pad with the average of other days if there are not enough days
        if len(prices) < 30:
            pad_width = 30 - len(prices)
            avg = np.mean(prices) if prices else 0  # Handling case with empty prices list
            prices = np.pad(prices, (pad_width, 0), 'constant', constant_values=(avg, avg))

        # Calculate y using the 'Prev Adj Close' column for comparison
        y = 1 if (pd.notnull(row['Prev Adj Close']) and row['Adj Close'] > row['Prev Adj Close']) else 0
        
        # Append to final_data_with_embeddings
        final_data_with_embeddings.append({
            'date': row['Date'],
            'prices': prices,
            'headlines': embeddings,
            'y': y
        })

# Convert final_data_with_embeddings to DataFrame for easier handling/manipulation
final_df_with_embeddings = pd.DataFrame(final_data_with_embeddings)

# Display the structure of the final DataFrame
print(final_df_with_embeddings.head())

# Display the first few rows to verify the structure
# final_df.head()

# Save the final_df to a csv file
final_df_with_embeddings.to_csv('data/final_df_with_embeddings.csv', index=False)
