# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

from pathlib import Path
import os
import re
import html
import string
import pickle

import unicodedata


import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer


from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
import nltk

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

def recommend_user_item(user_id):
    
    user_id_model = user_enc.transform([user_id])
    quantity = np.array([1])  # Placeholder value for quantity
    total_price = np.array([1])  # Placeholder value for total price
    
    products = df[['StockCode','Description_Cleaned']]
    products = products.drop_duplicates(subset=['StockCode'])
    item_id_model = item_enc.transform(products['StockCode'])
    
    # Repeat the user features for all items
    user_ids = np.repeat(user_id_model, products.shape[0])
    quantities = np.repeat(quantity, products.shape[0])
    total_prices = np.repeat(total_price, products.shape[0])
    
    max_sequence_length =  products['Description_Cleaned'].str.len().max()  # Maximum length of the item description sequence

    text_seq = tokenizer.texts_to_sequences(products['Description_Cleaned'].values)
    text_seq = pad_sequences(text_seq, maxlen=max_sequence_length)
    
    predictions = loaded_model.predict([user_ids, item_id_model, quantities,total_prices,text_seq])
    y_pred_classes = np.argmax(predictions, axis=1)
    products['relevancy'] = y_pred_classes
    items = list(products.sort_values(by='relevancy',ascending=False)[:10]['StockCode'])
    
    return items

def get_similar_items(item_id):
    item_id_encoded = item_enc.transform([item_id])[0]
    # Retrieve the embedding for the given item
    item_embedding = loaded_model.get_layer('item_embedding_LUT').get_weights()[0][item_id_encoded]
    # Compute the cosine similarities between the item embedding and all other item embeddings
    similarities = np.dot(loaded_model.get_layer('item_embedding_LUT').get_weights()[0], item_embedding)
    # Get the indices of the top 5 most similar items
    top_indices = np.argsort(similarities)[::-1][:10]
    simmilar_products = list(item_enc.inverse_transform(top_indices))
    return simmilar_products

def remove_special_chars(text):
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x1))


def remove_non_ascii(text):
    """Remove non-ASCII characters from list of tokenized words"""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def to_lowercase(text):
    return text.lower()



def remove_punctuation(text):
    """Remove punctuation from list of tokenized words"""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def replace_numbers(text):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    return re.sub(r'\d+', '', text)


def remove_whitespaces(text):
    return text.strip()


def remove_stopwords(words, stop_words):
    """
    :param words:
    :type words:
    :param stop_words: from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    or
    from spacy.lang.en.stop_words import STOP_WORDS
    :type stop_words:
    :return:
    :rtype:
    """
    return [word for word in words if word not in stop_words]


def stem_words(words):
    """Stem words in text"""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]

def lemmatize_words(words):
    """Lemmatize words in text"""

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def lemmatize_verbs(words):
    """Lemmatize verbs in text"""

    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word, pos='v') for word in words])

def text2words(text):
  return word_tokenize(text)

def normalize_text( text):
    text = remove_special_chars(text)
    text = remove_non_ascii(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_numbers(text)
    words = text2words(text)
    stop_words = stopwords.words('english')
    words = remove_stopwords(words, stop_words)
    #words = stem_words(words)# Either stem or lemmatize
    #words = lemmatize_words(words)
    #words = lemmatize_verbs(words)

    return ' '.join(words)

def normalize_corpus(corpus):
  return [normalize_text(t) for t in corpus]


item_enc = LabelEncoder()
item_enc.classes_ = np.load('../encoder/item_enc_1.npy',allow_pickle=True)
user_enc = LabelEncoder()
user_enc.classes_ = np.load('../encoder/user_enc_1.npy',allow_pickle=True)

# Load tokenizer
with open('../encoder/text_tokenizer_1.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    
# load model
loaded_model = load_model('../models/model_3_text_drop.h5')

# Load the dataset
df = pd.read_csv('../data/preprocessed_data_1.csv')
df['Description_Cleaned'] = normalize_corpus(df['Description'])