# clean data from train.csv
import pandas as pd
import csv
from datetime import datetime

csv_file_path = 'csv\clean_train.csv'
sorted_csv_file_path = 'clean_train_max25.csv'

# Load the CSV data into a DataFrame
df = pd.read_csv(csv_file_path, encoding='utf-8')

def sort_by_date(data):
    # Assuming your date column is named 'date' and the headline column is named 'headline'
    # If they have different names, replace 'date' and 'headline' with the actual column names
    try:
        # Convert the 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
    except KeyError:
        # If there's no 'date' column, it will raise a KeyError
        print("Error: No 'date' column found in the CSV file.")
        exit()
    except ValueError as e:
        # Handle rows with incorrect date format; for simplicity, let's drop these rows
        print(f"ValueError encountered: {e}")
        print("Rows with incorrect date format will be dropped.")
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        # Drop rows with NaT in 'date' column (where conversion failed)
        df = df.dropna(subset=['date'])

    # Select only the columns of interest
    df_cleaned = df[['headline', 'date']]

    # Sort by 'date' by latest to oldest
    df_sorted = df_cleaned.sort_values(by='date', ascending=False)

    # Save the cleaned and sorted data to a new CSV
    df_sorted.to_csv(sorted_csv_file_path, index=False, encoding='utf-8')


def max_25_news_per_day(data):

    df = pd.read_csv(data, encoding='utf-8')
    # We only need 2018 - 2020 data
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].dt.year >= 2018]
    # Group by date and select the first 25 rows of each group
    df = df.groupby('date').head(25)
    df.to_csv('clean_train_max25.csv', index=False, encoding='utf-8')

max_25_news_per_day(csv_file_path)
    