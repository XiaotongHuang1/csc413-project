from openai import OpenAI
import numpy as np
import pandas as pd

client = OpenAI(api_key='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')

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)


def embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small", input=text, encoding_format="float"
    )
    cut_dim = response.data[0].embedding[:256]
    norm_dim = normalize_l2(cut_dim)
    return norm_dim


def loop_csv(csv_file):
    results = []
    df = pd.read_csv(csv_file)[:]
    # add progress bar
    import tqdm
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        # Get headline
        headline = row['headline']
        # Get embedding
        emb = embedding(headline)
        # Append to results
        results.append({
            'headline': headline,
            'embedding': emb,
            'date': row['date']
        })

    # write to a new csv
    new_df = pd.DataFrame(results)
    new_df.to_csv('dataset_tmp.csv', index=False)


loop_csv('csv\clean_train_max25.csv')

# response = client.embeddings.create(
#     model="text-embedding-3-small", input="Testing 123", encoding_format="float"
# )

# cut_dim = response.data[0].embedding[:256]
# norm_dim = normalize_l2(cut_dim)

# print(norm_dim)