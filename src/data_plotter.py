from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def data_importer(filename):
    pandas_df = pd.read_csv(filename)
    return pandas_df


def plot_hist(pandas_df, column, bin_width):
    data_to_plot = pandas_df[column].dropna()
    df_range = data_to_plot.quantile([0.025, 0.975])
    np_data = data_to_plot.to_numpy()
    bins = np.arange(df_range[0.025], df_range[0.975] + bin_width, bin_width)
    plt.hist(np_data, bins=bins)
    plt.show()


df = data_importer('../data/DJI_99to23_prices_percentage_change.csv')
plot_hist(df, 'Ratio', 0.001)