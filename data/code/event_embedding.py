from openai import OpenAI
import numpy as np
import pandas as pd
import threading
from queue import Queue
import tqdm

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


# def loop_csv(csv_file):
#     results = []
#     df = pd.read_csv(csv_file)[:]
#     # add progress bar
#     import tqdm
#     for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
#         # Get headline
#         headline = row['headline']
#         # Get embedding
#         emb = embedding(headline)
#         # Append to results
#         results.append({
#             'headline': headline,
#             'embedding': emb,
#             'date': row['date']
#         })

#     # write to a new csv
#     new_df = pd.DataFrame(results)
#     new_df.to_csv('dataset_99to23.csv', index=False)


# loop_csv('csv\clean_train_max25.csv')



def process_chunk(queue, results, start, end, df):
    local_results = []

    for index, row in tqdm.tqdm(df.iloc[start:end].iterrows(), total=end - start):
        # Simulate processing
        headline = row['headline']
        emb = embedding(headline)  # Assuming embedding is a function defined elsewhere
        local_results.append({
            'headline': headline,
            'embedding': emb,
            'date': row['date']
        })
    queue.put(local_results)

def loop_csv_multithreaded(csv_file, num_threads=24):
    df = pd.read_csv(csv_file)
    chunk_size = len(df) // num_threads
    threads = []
    queue = Queue()
    results = []

    # Split work among threads
    for i in range(num_threads):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_threads - 1 else len(df)
        thread = threading.Thread(target=process_chunk, args=(queue, results, start, end, df))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Collect results
    while not queue.empty():
        results.extend(queue.get())

    # Write to a new CSV
    new_df = pd.DataFrame(results)
    new_df.to_csv('dataset_embedding_99to23.csv', index=False)

# Example usage
loop_csv_multithreaded('data/sorted_cnbc.csv')