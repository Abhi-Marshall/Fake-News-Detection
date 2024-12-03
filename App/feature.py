import numpy as np  # linear algebra
import pandas as pd  # data processing

import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Define stop words
stop_words = set(stopwords.words('english'))

def get_all_query(title, author, text):
    total = title + author + text
    total = [total]
    return total

def remove_punctuation_stopwords_lemma(sentence):
    filter_sentence = ''
    lemmatizer = WordNetLemmatizer()
    # Remove punctuation using regex
    sentence = re.sub(r'[^\w\s]', '', sentence)  # Removing punctuation
    # Split the sentence into words using split method
    words = sentence.split()  # Tokenization using split
    words = [w for w in words if not w in stop_words]  # Removing stopwords
    for word in words:
        filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(word)).lower()
    return filter_sentence
