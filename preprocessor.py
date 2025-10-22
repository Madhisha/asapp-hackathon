# preprocessor.py
import re

def clean_text(text):
    """
    Clean user input text
    """
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text
