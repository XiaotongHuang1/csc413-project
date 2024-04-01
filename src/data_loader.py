import numpy as np
import pandas as pd
import ast

def merge_dataframes_single_headline(df1, df2):
    df1 = pd.read_csv(df1)
    df2 = pd.read_csv(df2)

    df2 = df2.drop('pre30', axis=1)
    df2 = df2.drop('post7', axis=1)
    df2 = df2.drop('headline', axis=1)
    # df1['date'] = pd.to_datetime(df1['Date'].astype(str).dt.strftime('%Y-%m-%d'))
    df1['date'] = df1['Date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
    df2['date'] = pd.to_datetime(df2['date'], errors='coerce')
    df3 = pd.merge(df1, df2, on='date', how='inner')
    df3 = df3.drop('date', axis=1)
    df3 = df3.drop('Date', axis=1)

    df3.to_csv('../data/final_dataset/finetune_data.csv')


# merge_dataframes_single_headline('../data/final_dataset/filtered_unique_headlines_sorted_64.csv',
#                                  '../data/final_dataset/merged_file.csv')


def get_label(df, daily_price):
    df = pd.read_csv(df)
    df_dp = pd.read_csv(daily_price)

    df_dp['date'] = df_dp['Date']
    df_dp['daily_price'] = df_dp['Adj Close']
    df_dp = df_dp.drop('Date', axis=1)
    df_dp = df_dp.drop('Adj Close', axis=1)
    df_dp = df_dp.drop('Ratio', axis=1)

    df = pd.merge(df, df_dp, on='date')

    df['post7'] = df['post7'].apply(lambda x: ast.literal_eval(x))
    df['post7avg'] = df['post7'].apply(lambda x: sum(x) / len(x))
    df['ratio'] = df.apply(lambda x: x['post7avg'] / x['daily_price'], axis=1)

    def _check_label(score, lower, higher):
        if score < lower:
            return 2
        elif score > higher:
            return 1
        else:
            return 0

    lower_bound, higher_bound = df['ratio'].quantile([0.35, 0.65])
    print(lower_bound, higher_bound)
    df['label'] = df['ratio'].apply(lambda x: _check_label(x, lower_bound, higher_bound))

    df.to_csv('../data/final_dataset/labeled_merged_data.csv')


# get_label("../data/final_dataset/merged_file.csv",
#           "../data/final_dataset/DJI_99to23_prices_percentage_change.csv")


def update_label(df, df_l):
    df = pd.read_csv(df)
    df_l = pd.read_csv(df_l)

    df = df.drop('y', axis=1)

    df_l = df_l.drop('pre30', axis=1)
    df_l = df_l.drop('post7', axis=1)
    df_l = df_l.drop('headline', axis=1)
    df_l = df_l.drop('daily_price', axis=1)
    df_l = df_l.drop('post7avg', axis=1)
    df_l = df_l.drop('ratio', axis=1)
    df_l = df_l.drop('y', axis=1)
    df_l['y'] = df_l['label']
    df_l = df_l.drop('label', axis=1)
    df_l = df_l.drop('Unnamed: 0', axis=1)

    df = pd.merge(df, df_l, on='date')

    df.to_csv('../data/final_dataset/updated_dataset.csv', index=False)

    df = df.drop('embedding', axis=1)
    df = df.drop('pre128', axis=1)
    df.to_csv('../data/final_dataset/clean_dataset.csv', index=False)


update_label('../data/final_dataset/dataset.csv', '../data/final_dataset/labeled_merged_data.csv')