import os
import json
import pandas as pd
import logging
import re
import faiss
import numpy as np
from typing import Union, List, Tuple
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
import openai
from nltk.stem import WordNetLemmatizer


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Sentence Transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

def load_data(file_path: str) -> Union[pd.DataFrame, list, None]:
    """Load data from CSV, JSON, or TXT formats."""
    try:
        if file_path.endswith(".csv"):
            data = pd.read_csv(file_path)
            logging.info("CSV file loaded successfully.")
            return data
        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logging.info("JSON file loaded successfully.")
            return data
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.readlines()
            logging.info("TXT file loaded successfully.")
            return data
        else:
            logging.error("Unsupported file format. Only CSV, JSON, and TXT are allowed.")
            return None
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        return None

def clean_text(text: str) -> str:
    """Preprocess text: lowercasing, remove special characters, stopwords filtering, lemmatization."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in ENGLISH_STOP_WORDS]
    return " ".join(filtered_words)

def preprocess_dataframe(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    """Apply text preprocessing to specified columns of a DataFrame."""
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(clean_text)
            logging.info(f"Preprocessed column: {col}")
        else:
            logging.warning(f"Column {col} not found in DataFrame.")
    return df

def preprocess_text_lines(lines: list) -> list:
    """Preprocess each line of a text file."""
    return [clean_text(line) for line in lines]

def embed_texts(texts: List[str]) -> np.ndarray:
    """Generate text embeddings using a Sentence Transformer model."""
    return np.array(embedding_model.encode(texts, convert_to_numpy=True))

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS index for efficient similarity search."""
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve_similar_documents(query, index, texts, top_k=3, threshold=0.5):
    """Retrieve the most similar documents based on query embedding similarity."""
    
    # If query is empty, return no results
    if not query.strip():
        return []

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Compute query embedding
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Perform FAISS search
    D, I = index.search(query_embedding, top_k)  # D = distances (similarity scores), I = indices

    # Debugging: Print raw FAISS results
    print("Query:", query)
    print("FAISS Retrieved Indices:", I)
    print("FAISS Similarity Scores:", D)

    # Normalize similarity scores (since FAISS uses L2 distance by default)
    similarities = 1 / (1 + D)  # Convert distances to similarity scores

    # Filter results based on threshold
    results = []
    for i, score in zip(I[0], similarities[0]):
        if i != -1 and score >= threshold:  # Ensure valid index and similarity threshold met
            results.append(texts[i])

    return results


def generate_response(query: str, documents: List[str]) -> str:
    """Generate a response using OpenAI's GPT API."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Query: {query}\n\nRelevant Documents:\n{chr(10).join(documents)}\n\nAnswer:"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are an expert in document retrieval."},
                      {"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "Error generating response."

def process_file(file_path: str, text_columns: list = None) -> Tuple[Union[pd.DataFrame, list, None], List[str], np.ndarray, faiss.IndexFlatL2]:
    """Main function to load, preprocess, and create embeddings for a file."""
    data = load_data(file_path)
    
    if isinstance(data, pd.DataFrame) and text_columns:
        data = preprocess_dataframe(data, text_columns)
        texts = data[text_columns[0]].tolist()
    elif isinstance(data, list):  # JSON or TXT
        texts = preprocess_text_lines(data) if isinstance(data[0], str) else [clean_text(entry[text_columns[0]]) for entry in data if text_columns[0] in entry]
    else:
        return None, [], None, None
    
    embeddings = embed_texts(texts)
    index = build_faiss_index(embeddings)
    return data, texts, embeddings, index

if __name__ == "__main__":
    file_path = "documents.csv"  # Change this to your dataset file
    text_columns = ["content"]  # Specify the column containing text data
    
    data, texts, embeddings, index = process_file(file_path, text_columns)
    
    if index:
        query = "What is lung cancer?"
        relevant_docs = retrieve_similar_documents(query, index, texts)
        response = generate_response(query, relevant_docs)
        print("\nGenerated Response:\n", response)
