import numpy as np
import pandas as pd
import ast
import time
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import torch
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from transformers.pipelines.pt_utils import KeyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "")
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

batch = 128
input_name = '../data/final_dataset/filtered_unique_headlines_sorted_64.csv'


def generate_finbert_data(filename, batch_size):
    df = pd.read_csv(filename)
    # print(len(df))
    nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer, device=device, batch_size=batch_size)
    df['prediction'] = ""
    df['score'] = ""
    df['index'] = df.index

    count = 0
    warnings.filterwarnings("ignore")
    for _, df_group in df.groupby(df.index // batch_size):
        try:
            headlines = df_group["Headline"].tolist()
            # print(headlines)
            results = nlp(headlines)
            # print(results)
            start_index = df_group.iloc[0]['index']
            end_index = df_group.iloc[batch_size - 1]['index']
            # print(start_index, end_index)

            predictions = [x["label"] for x in results]
            scores = [x["score"] for x in results]

            df.loc[start_index:end_index, 'prediction'] = predictions
            df.loc[start_index:end_index, 'score'] = scores

            count += 1
            if count % 100 == 0:
                print(f"Processed {count} of batches, total number of rows: {count * batch_size}")
        except Exception as e:
            print(f"Error processing batch {e}")
            continue
    print(df)
    df.to_csv('../data/final_dataset/finbert_predict.csv')


def generate_finbert_data_merged(filename):
    df = pd.read_csv(filename)
    nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer, device=device, batch_size=32)
    df['headline'] = df['headline'].apply(lambda x: ast.literal_eval(x))

    df['prediction'] = ""
    df['index'] = df.index

    count = 0
    for index, row in df.iterrows():
        try:
            result = nlp(row['headline'])
            predictions = [x['label'] for x in result]
            scores = [str(x['score']) for x in result]

            s_predictions = '[' + ",".join(predictions) + ']'
            s_scores = '[' + ",".join(scores) + ']'

            df.loc[index, 'predictions'] = s_predictions
            df.loc[index, 'scores'] = s_scores
            count += 1
            if count % 100 == 0:
                print(count)

        except Exception as e:
            print(e)
            continue
    df.to_csv('../data/final_dataset/merged_finbert_predict.csv')


#
generate_finbert_data(input_name, batch)
# generate_finbert_data_merged('../data/final_dataset/merged_file.csv')
