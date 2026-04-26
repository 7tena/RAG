import numpy as np
import os
from utils import (
    display_widget,
    plot_vectors
)
# Distance formulas. 
# In this ungraded lab, distance formulas are implemented here. In future assignments, you will import functions from specialized libraries.
def cosine_similarity(v1, array_of_vectors):
    """
    Compute the cosine similarity between a vector and an array of vectors.
    
    Parameters:
    v1 (array-like): The first vector.
    array_of_vectors (array-like): An array of vectors or a single vector.

    Returns:
    list: A list of cosine similarities between v1 and each vector in array_of_vectors.
    """
    # Ensure that v1 is a numpy array
    v1 = np.array(v1)
    # Initialize a list to store similarities
    similarities = []
    
    # Check if array_of_vectors is a single vector
    if len(np.shape(array_of_vectors)) == 1:
        array_of_vectors = [array_of_vectors]
    
    # Iterate over each vector in the array
    for v2 in array_of_vectors:
        # Convert the current vector to a numpy array
        v2 = np.array(v2)
        # Compute the dot product of v1 and v2
        dot_product = np.dot(v1, v2)
        # Compute the norms of the vectors
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        # Compute the cosine similarity and append to the list
        similarity = dot_product / (norm_v1 * norm_v2)
        similarities.append(similarity)
    return [float(x) for x in similarities]

def euclidean_distance(v1, array_of_vectors):
    """
    Compute the Euclidean distance between a vector and an array of vectors.
    
    Parameters:
    v1 (array-like): The first vector.
    array_of_vectors (array-like): An array of vectors or a single vector.

    Returns:
    list: A list of Euclidean distances between v1 and each vector in array_of_vectors.
    """
    # Ensure that v1 is a numpy array
    v1 = np.array(v1)
    # Initialize a list to store distances
    distances = []
    
    # Check if array_of_vectors is a single vector
    if len(np.shape(array_of_vectors)) == 1:
        array_of_vectors = [array_of_vectors]
    
    # Iterate over each vector in the array
    for v2 in array_of_vectors:
        # Convert the current vector to a numpy array
        v2 = np.array(v2)
        # Check if the input arrays have the same shape
        if v1.shape != v2.shape:
            raise ValueError(f"Shapes don't match: v1 shape: {v1.shape}, v2 shape: {v2.shape}")
        # Calculate the Euclidean distance and append to the list
        dist = np.sqrt(np.sum((v1 - v2) ** 2))
        distances.append(dist)
    return [float(x) for x in distances]


# Example 
v1 = [1, 2]
v2 = [1, 1]
array_v = [[3, 2], [5, 6]]
cosine_v1_v2 = cosine_similarity(v1, v2)
cosine_v1_array_v = cosine_similarity(v1, array_v)
euclidean_v1_v2 = euclidean_distance(v1, v2)
euclidean_v1_array_v = euclidean_distance(v1, array_v)
print(f"Cosine Similarity between v1 and v2: {cosine_v1_v2}")
print(f"Cosine Similarities between v1 and array_v: {cosine_v1_array_v}")
print(f"Euclidean Distance between v1 and v2: {euclidean_v1_v2}")
print(f"Euclidean Distances between v1 and array_v: {euclidean_v1_array_v}")

#Loading the model
from sentence_transformers import SentenceTransformer

# Load the pre-trained sentence transformer model
model_name =  "BAAI/bge-base-en-v1.5"


model = SentenceTransformer(os.path.join(os.environ['MODELS'],model_name))
# To get a string embedded, just pass it to the model.
res = model.encode("RAG is awesome")
print(res.shape)
# An array of strings can be passed, and the output will be an array of vectors, each with 768 dimensions.
model.encode(['apple', 'car'])
# Print the first 100 elements of the embedding.
print(res[:100])

# Embeddings in practice:
words = ['apple', 'car', 'fruit', 'automobile', 'love', 'sentiment']
vectorized_words = model.encode(words)
word = 'apple'
print(f"{word}:")
for i, w in enumerate(words):
    # Get the vectorized word for the word defined above
    vectorized_word = vectorized_words[words.index(word)]
    print(f"\t{w}:\t\tCosine Similarity: {cosine_similarity(vectorized_word, vectorized_words[i])[0]:.4f}")
print("\n\n\n")
for i, w in enumerate(words):
    # Get the vectorized word for the word defined above
    vectorized_word = vectorized_words[words.index(word)]
    print(f"\t{w}:\t\tEuclidean Distance: {euclidean_distance(vectorized_word, vectorized_words[i])[0]:.4f}")

def retrieve_relevant(query, documents, metric='cosine_similarity'):
    """
    Retrieves and ranks documents based on their similarity to a given query using the specified metric.
    
    Parameters:
    query (str): The query string for which relevant documents are to be retrieved.
    documents (list of str): A list of documents to be compared against the query.
    metric (str, optional): The similarity measurement metric to be used. It supports 'cosine_similarity'
                            and 'euclidean'. Defaults to 'cosine_similarity'.
    
    Returns:
    list of tuples: A list of tuples where each tuple contains a document and its similarity or distance
                    score with respect to the query. The list is sorted based on these scores, with
                    descending order for 'cosine_similarity' and ascending order for 'euclidean'.
    """
    query_emb = model.encode(query)
    documents_emb = model.encode(documents)
    vals = []

    if metric == 'cosine_similarity':
        distances = cosine_similarity(query_emb, documents_emb)
        vals = [(doc, dist) for doc, dist in zip(documents, distances)]
        # Sort in descending order
        vals.sort(reverse=True, key=lambda x: x[1])
        
    elif metric == 'euclidean':
        distances = euclidean_distance(query_emb, documents_emb)
        vals = [(doc, dist) for doc, dist in zip(documents, distances)]
        # Sort in ascending order
        vals.sort(key=lambda x: x[1])
        
    return vals

documents = [
    "Mt. Fuji is a breathtaking place to explore during autumn.",
    "Santorini offers stunning views to admire during spring.",
    "Banff National Park is a picturesque destination to visit in the summer.",
    "The Great Wall of China is a spectacular site to experience during winter.",
    "The fjords of Norway are a magical place to cruise through in the spring.",
    "Prague is an enchanting city to wander through in winter.",
    "Kyoto's cherry blossoms create a beautiful scene to witness during spring.",
    "Marrakech offers vibrant markets and culture to enjoy in the fall.",
    "The Maldives are a paradisiacal getaway to savor during summer.",
    "The Christmas markets in Vienna are a festive delight to explore in winter."
]

query = "Suggest to me great places to visit in Asia."
score = retrieve_relevant(query, documents, metric='cosine_similarity')
score = retrieve_relevant(query, documents, metric='euclidean')

# Embeddings and input size
big_text = open("large_text.txt").read()
# Entire text
big_text_embedding = model.encode(big_text)

# Text with fewer characters
big_text_embedding_few_characters = model.encode(big_text[:3000])

# Checking if they are the same
np.array_equal(big_text_embedding, big_text_embedding_few_characters)
