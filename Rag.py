import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import ssl
import numpy as np
import faiss
from langchain_community.embeddings import OllamaEmbeddings

# Ignore SSL certificate errors
ssl._create_default_https_context = ssl._create_unverified_context
embedding = OllamaEmbeddings(model="mxbai-embed-large")

# Download required NLTK resources
nltk_data_path = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_path):
    nltk.download('punkt_tab', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)
else:
    print("NLTK data already available, skipping download.")
    

def preprocess_text_nltk(text):
    """Preprocess text: lowercase, remove punctuation, tokenize, remove stopwords, apply stemming."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return ' '.join(tokens)

def read_file(file_path):
    """Read a text file and return its content."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks with overlap."""
    if overlap >= chunk_size:
        raise ValueError("Overlap should be smaller than chunk size.")
    words = word_tokenize(text)
    chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks

def search_faiss(query, index, top_k=3):
    """Search FAISS index for the most similar embeddings."""
    query_embedding = embedding.embed_query(query)
    query_embedding_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    if query_embedding_np.shape[1] != embedding_dim:
        raise ValueError(f"Query embedding dimension mismatch: Expected {embedding_dim}, got {query_embedding_np.shape[1]}")
    D, I = index.search(query_embedding_np, top_k)
    return D[0], I[0]

if __name__ == "__main__":
    file_path = "speech.txt"
    speech = read_file(file_path)
    cleaned_text = preprocess_text_nltk(speech)
    chunks = split_text_into_chunks(cleaned_text)
    
    document_embeddings = [embedding.embed_query(chunk) for chunk in chunks]
    embeddings_np = np.array(document_embeddings, dtype=np.float32)
    
    embedding_dim = 1024
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)
    
    query = "world peace planted upon what."
    distances, indices = search_faiss(query, index)
    
    print("Closest matching chunks:")
    for idx, distance in zip(indices, distances):
        if idx < len(chunks):
            print(f"- Chunk {idx}: (Distance: {distance:.4f}) {chunks[idx]}")
        else:
            print(f"- Index {idx} is out of bounds.")
