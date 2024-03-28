import pandas as pd

# Load the dataset
df = pd.read_csv('data/DJI_99to23_prices.csv')

# Calculate the percentage change
df['Ratio'] = df['Adj Close'].pct_change() + 1


min_ratio = df['Ratio'].min()
max_ratio = df['Ratio'].max()
mean_ratio = df['Ratio'].mean()
std_dev_ratio = df['Ratio'].std()
quartiles = df['Ratio'].quantile([0.25, 0.5, 0.75]).to_dict()

# Display the statistics
print(f"Minimum Ratio: {min_ratio}")
print(f"Maximum Ratio: {max_ratio}")
print(f"Mean Ratio: {mean_ratio}")
print(f"Standard Deviation of Ratio: {std_dev_ratio}")
print(f"Quartiles: {quartiles}")

# Save the dataset
# df.to_csv('data/DJI_99to23_prices_percentage_change.csv', index=False)
