import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, Trainer, BertForSequenceClassification, TrainingArguments

df = pd.read_csv('../data/final_data.csv') ## use your own customized dataset
df['headlines'] = df['headlines'].apply(lambda x: ast.literal_eval(x))
df['headlines'] = df['headlines'].apply(lambda x: ' [SEP] '.join(x))
df['prices'] = df['prices'].apply(lambda x: x.replace("[", "").replace("]", ""))
# df['input'] = df.apply(lambda row: row['headlines'], axis=1)

df = df.drop('date', axis=1)
df = df.drop('prices', axis=1)
# df = df.drop('headlines', axis=1)

df.to_csv('../data/clean_input.csv', index=False)

print(df.columns)
print(df.head())

df_train, df_test, = train_test_split(df, stratify=df['y'], test_size=0.1, random_state=42)
df_train, df_val = train_test_split(df_train, stratify=df_train['y'],test_size=0.1, random_state=42)
print(df_train.shape, df_test.shape, df_val.shape)

# finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
# tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')