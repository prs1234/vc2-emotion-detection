import os
import re
import numpy as np
import pandas as pd
import nltk
import string
import logging
from typing import Any, Union
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists

logging.basicConfig(
    filename='logs/data_preprocessing.log',  # <--- Save logs to file
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_nltk_resources() -> None:
    try:
        nltk.download('wordnet')
        nltk.download('stopwords')
        logging.info("NLTK resources downloaded")
    except Exception as e:
        logging.error(f"Error downloading NLTK resources: {e}")
        raise

def lemmatization(text: str) -> str:
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(y) for y in text]
        return " ".join(text)
    except Exception as e:
        logging.error(f"Lemmatization error: {e}")
        return text

def remove_stop_words(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        Text = [i for i in str(text).split() if i not in stop_words]
        return " ".join(Text)
    except Exception as e:
        logging.error(f"Stop word removal error: {e}")
        return text

def removing_numbers(text: str) -> str:
    try:
        text = ''.join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        logging.error(f"Number removal error: {e}")
        return text

def lower_case(text: str) -> str:
    try:
        text = text.split()
        text = [y.lower() for y in text]
        return " ".join(text)
    except Exception as e:
        logging.error(f"Lowercase conversion error: {e}")
        return text

def removing_punctuations(text: str) -> str:
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "", )
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        logging.error(f"Punctuation removal error: {e}")
        return text

def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logging.error(f"URL removal error: {e}")
        return text

def remove_small_sentences(df: pd.DataFrame) -> None:
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
        logging.info("Small sentences removed")
    except Exception as e:
        logging.error(f"Error removing small sentences: {e}")

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.content = df.content.apply(lambda content: lower_case(content))
        df.content = df.content.apply(lambda content: remove_stop_words(content))
        df.content = df.content.apply(lambda content: removing_numbers(content))
        df.content = df.content.apply(lambda content: removing_punctuations(content))
        df.content = df.content.apply(lambda content: removing_urls(content))
        df.content = df.content.apply(lambda content: lemmatization(content))
        logging.info("Text normalization completed")
        return df
    except Exception as e:
        logging.error(f"Error normalizing text: {e}")
        return df

def normalized_sentence(sentence: str) -> str:
    try:
        sentence = lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = removing_numbers(sentence)
        sentence = removing_punctuations(sentence)
        sentence = removing_urls(sentence)
        sentence = lemmatization(sentence)
        return sentence
    except Exception as e:
        logging.error(f"Error normalizing sentence: {e}")
        return sentence

def load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Loaded data from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        raise

def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logging.info(f"Saved data to {filepath}")
    except Exception as e:
        logging.error(f"Error saving data to {filepath}: {e}")
        raise

def main() -> None:
    download_nltk_resources()
    train_data = load_data("data/raw/train.csv")
    test_data = load_data("data/raw/test.csv")
    train_data = normalize_text(train_data)
    test_data = normalize_text(test_data)
    save_data(train_data, "data/processed/train.csv")
    save_data(test_data, "data/processed/test.csv")

if __name__ == "__main__":
    main()
