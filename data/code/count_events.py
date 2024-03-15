import pandas as pd
import pandas_market_calendars as mcal

# 计数每日新闻数量
def count_daily_news(data):
    # 读取 CSV 文件
    df = pd.read_csv(data, encoding='utf-8')

    # 确保 'date' 列是 datetime 类型
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 筛选出 2020 年的数据
    # df_2020 = df[df['date'].dt.year == 2020]
    # do same for each year

    for year in range(2010, 2021):
        count_num = 25
        df_year = df[df['date'].dt.year == year]
        daily_counts = df_year.groupby(df_year['date'].dt.date).size()
        dates_less_than_5_news = daily_counts[daily_counts < count_num].index.tolist()
        print(f"在 {year} 年，以下日期的新闻数量 < {count_num} 条：")
        for date in dates_less_than_5_news:
            print(date)

# 计算每日新闻数量
# csv_file_path = 'data/clean_train_10years_max25.csv'
# count_daily_news(csv_file_path)