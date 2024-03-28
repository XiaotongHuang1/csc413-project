import numpy as np
import pandas as pd
import sys

# Set print options to ensure all elements of an array are printed
np.set_printoptions(threshold=sys.maxsize)


def reduce_dim(embedding, expected_dim=128):
    """
    Reduces the dimensionality of the embedding from 256 to expected_dim.
    :param embedding_str: The string representation of the embedding.
    :param expected_dim: The expected dimensionality of the embedding.
    :return: The reduced dimensionality embedding.
    """
    # Ensure the embedding has the expected dimensionality
    assert embedding.shape[0] == 256, f"Embedding has unexpected dimensionality: {embedding.shape[0]}"

    # Reshape the embedding to the expected dimensionality
    embedding = embedding.reshape((expected_dim, 256 // expected_dim))

    # Sum the values in each row to reduce the dimensionality
    embedding = np.sum(embedding, axis=1)

    return embedding


# Load the data from the csv files
headline_embeddings_data = pd.read_csv('data/final_dataset/sorted_embedding_headlines_processed.csv')

# Reduce the dimensionality of the embeddings
headline_embeddings_data['embedding'] = headline_embeddings_data['embedding'].apply(lambda x: reduce_dim(np.fromstring(x[1:-1], sep=' ')))

# Ensure data is sorted by date
headlines_data = headline_embeddings_data.sort_values(by='Date').reset_index(drop=True)

# Output the processed data to a new csv file
headlines_data.to_csv('data/final_dataset/sorted_embedding_headlines_processed_128.csv', index=False)