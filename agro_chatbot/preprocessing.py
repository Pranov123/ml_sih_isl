import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import HuggingFaceEmbeddings
from utils.agro_dict import AGRO_DICT

# Initialize the HuggingFace embeddings model
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to compute embeddings for a list of words and store them in a pickle file
def save_embeddings_to_pickle(words, pickle_file="embeddings_agro.pkl"):
    """
    Generate embeddings for a list of words and save them to a pickle file.
    """
    word_embeddings = {word: embeddings_model.embed_query(word) for word in words}
    with open(pickle_file, "wb") as f:
        pickle.dump(word_embeddings, f)
    print(f"Embeddings saved to {pickle_file}")

# Function to find the most similar word given an input word
def find_most_similar_word(input_word, pickle_file="embeddings_agro.pkl", threshold=0.7):
    """
    Find the most similar word to the input word using cosine similarity.

    :param input_word: The word for which to find the most similar word.
    :param pickle_file: Path to the pickle file containing saved embeddings.
    :param threshold: Cosine similarity threshold.
    :return: Most similar word if similarity > threshold, else None.
    """
    # Load embeddings from pickle file
    with open(pickle_file, "rb") as f:
        word_embeddings = pickle.load(f)

    # Generate embedding for the input word
    input_embedding = embeddings_model.embed_query(input_word)

    # Compute cosine similarities
    similarities = {
        word: cosine_similarity([input_embedding], [embedding])[0][0]
        for word, embedding in word_embeddings.items()
    }

    # Find the most similar word
    most_similar_word = max(similarities, key=similarities.get)
    max_similarity = similarities[most_similar_word]

    if max_similarity >= threshold:
        return most_similar_word, max_similarity
    else:
        return None, max_similarity

# Example Usage
if __name__ == "__main__":
    # Define a list of words
    words = AGRO_DICT.keys()

    # Step 1: Save embeddings to a pickle file
    # save_embeddings_to_pickle(words)

    # Step 2: Find the most similar word to "grapefruit"
    input_word = "an electrode which is negative"
    similar_word, similarity = find_most_similar_word(input_word)
    if similar_word:
        print(f"The most similar word to '{input_word}' is '{similar_word}' with similarity {similarity:.2f}")
    else:
        print(f"No similar word found for '{input_word}' above the threshold.")