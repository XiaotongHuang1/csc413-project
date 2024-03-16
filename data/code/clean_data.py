# clean data from train.csv
import pandas as pd
import csv
from datetime import datetime

csv_file_path = 'data/final_dataset/embedding_headlines_processed.csv'
sorted_csv_file_path = 'data/final_dataset/sorted_embedding_headlines_processed.csv'


def sort_by_date(data):
    # Load the CSV data into a DataFrame
    df = pd.read_csv(data, encoding='utf-8')
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



def sort_by_date1(data, sorted_csv_file_path):
    try:
        # Attempt to read the CSV, skipping bad lines
        df = pd.read_csv(data, encoding='utf-8', on_bad_lines='skip')
        
        # Convert the 'Date' column to datetime format (assuming the format is '%Y%m%d')
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        
        # Select only the columns of interest, ensure column names match your CSV
        df_cleaned = df[['Date', 'embedding', 'headline']]  # Update column names as necessary
        
        # Sort by 'Date' in descending order
        df_sorted = df_cleaned.sort_values(by='Date', ascending=False)
        
        # Save the cleaned and sorted data to a new CSV
        df_sorted.to_csv(sorted_csv_file_path, index=False, encoding='utf-8')
    except KeyError as e:
        print(f"Error: Missing column in the CSV file. {e}")
    except ValueError as e:
        print(f"ValueError encountered: {e}. Rows with incorrect date format will be dropped.")



def sort_by_date_and_convert_date_format(input_file_path, output_file_path):
    # Load the CSV data into a DataFrame
    df = pd.read_csv(input_file_path, encoding='utf-8')

    try:
        # Convert the 'date' column to datetime format, including timezone
        # The 'date' column includes timezone information, hence '%Y-%m-%d %H:%M:%S%z'
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Convert datetime to the new format without timezone information
        # This step also effectively removes the timezone information
        df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Drop rows where date conversion was not possible (if any)
        df = df.dropna(subset=['date'])

        # Sort by 'date' by latest to oldest
        df_sorted = df.sort_values(by='date', ascending=True)

        # Save the sorted data with the converted date format to a new CSV
        df_sorted.to_csv(output_file_path, index=False, encoding='utf-8')

    except KeyError as e:
        print(f"Error: Missing expected column in the CSV file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def max_25_news_per_day(data):

    df = pd.read_csv(data, encoding='utf-8')
    # We only need 2010 - 2020 data
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].dt.year >= 2010]
    # if there don't have 25 news, drop the date
    df = df.groupby('date').filter(lambda x: len(x) >= 25)
    # Group by date and select the first 25 rows of each group
    df = df.groupby('date').head(25)
    df.to_csv('clean_train_max25.csv', index=False, encoding='utf-8')


def clean_data_from_file(data):
    # Load the CSV data into a DataFrame
    df = pd.read_csv(data, encoding='utf-8')
    
    # Convert 'Date' column to datetime format (assuming the format is YYYYMMDD)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    
    # Select only the 'Date' and 'Headline' columns
    df_filtered = df[['Date', 'Headline']]
    
    # Drop rows where 'Headline' is missing
    df_filtered = df_filtered.dropna(subset=['Headline'])
    
    # Ensure that the 'Headline' column does not contain empty strings
    df_filtered = df_filtered[df_filtered['Headline'].str.strip() != '']
    
    # Add a new column for the length of each 'Headline'
    df_filtered['HeadlineLength'] = df_filtered['Headline'].str.len()
    
    # Sort the DataFrame by 'Date' and 'HeadlineLength' in descending order
    df_sorted = df_filtered.sort_values(by=['Date', 'HeadlineLength'], ascending=[True, False])
    
    # We want to keep only the first 50 headlines for each date
    df_sorted = df_sorted.groupby('Date').head(50)
    
    # Drop the 'HeadlineLength' column as it's no longer needed after sorting
    df_sorted = df_sorted.drop(columns=['HeadlineLength'])
    
    # Save the processed DataFrame back to CSV
    df_sorted.to_csv(data.replace('.csv', '_processed.csv'), index=False, encoding='utf-8')


# clean_data_from_file('data/headlines.csv')

# sort_by_date_and_convert_date_format(csv_file_path, sorted_csv_file_path)
sort_by_date1(csv_file_path, sorted_csv_file_path)
# max_25_news_per_day(sorted_csv_file_path)
    