import os
import json
import pandas as pd
import logging
from typing import Union
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    """Preprocess text: lowercasing, remove special characters, stopwords filtering."""
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    words = text.split()
    filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS]  # Remove stopwords
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

def preprocess_json(data: list, text_keys: list) -> list:
    """Apply text preprocessing to specified keys in a JSON list."""
    for entry in data:
        for key in text_keys:
            if key in entry:
                entry[key] = clean_text(entry[key])
    logging.info("Preprocessed JSON data successfully.")
    return data

def preprocess_text_lines(lines: list) -> list:
    """Preprocess each line of a text file."""
    return [clean_text(line) for line in lines]

def process_file(file_path: str, text_columns: list = None) -> Union[pd.DataFrame, list, None]:
    """Main function to load and preprocess a file based on format."""
    data = load_data(file_path)
    
    if isinstance(data, pd.DataFrame) and text_columns:
        return preprocess_dataframe(data, text_columns)
    elif isinstance(data, list):  # JSON or TXT
        return preprocess_json(data, text_columns) if isinstance(data[0], dict) else preprocess_text_lines(data)
    return data
