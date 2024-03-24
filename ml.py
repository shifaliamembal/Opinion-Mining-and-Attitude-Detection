from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np # linear algebra
import pandas as pd # data processing
#import tweepy as tw #for accessing Twitter API

#For Preprocessing
import re    # RegEx for removing non-letter characters
import nltk  #natural language processing
#nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *

# For Building the model
from sklearn.model_selection import train_test_split
import tensorflow as tf

# # Load Tweet dataset
# df1 = pd.read_csv('Twitter_Data.csv')

# # Load Tweet dataset
# df2 = pd.read_csv('apple-twitter-sentiment-texts.csv')
# df2 = df2.rename(columns={'text': 'clean_text', 'sentiment':'category'})
# df2['category'] = df2['category'].map({-1: -1.0, 0: 0.0, 1:1.0})
# # Output first five rows

# # Load Tweet dataset
# df3 = pd.read_csv('finalSentimentdata2.csv')
# df3 = df3.rename(columns={'text': 'clean_text', 'sentiment':'category'})
# df3['category'] = df3['category'].map({'sad': -1.0, 'anger': -1.0, 'fear': -1.0, 'joy':1.0})
# df3 = df3.drop(['Unnamed: 0'], axis=1)
# # Output first five rows

# # Load Tweet dataset
# df4 = pd.read_csv('Tweets.csv')
# df4 = df4.rename(columns={'text': 'clean_text', 'airline_sentiment':'category'})
# df4['category'] = df4['category'].map({'negative': -1.0, 'neutral': 0.0, 'positive':1.0})
# df4 = df4[['category','clean_text']]
# # Output first five rows

# df = pd.concat([df1, df2, df3, df4], ignore_index=True)
# # Check for missing data
# df.isnull().sum()
# # drop missing rows
# df.dropna(axis=0, inplace=True)

# # Map tweet categories
# df['category'] = df['category'].map({-1.0:'Negative', 0.0:'Neutral', 1.0:'Positive'})

# df['length'] = df.clean_text.str.split().apply(len)

# df.drop(['length'], axis=1, inplace=True)

# def tweet_to_words(tweet):
#     ''' Convert tweet text into a sequence of words '''

#     # convert to lowercase
#     text = tweet.lower()
#     # remove non letters
#     text = re.sub(r"[^a-zA-Z0-9]", " ", text)
#     # tokenize
#     words = text.split()
#     # remove stopwords
#     words = [w for w in words if w not in stopwords.words("english")]
#     # apply stemming
#     words = [PorterStemmer().stem(w) for w in words]
#     # return list
#     return words

# # Apply data processing to each tweet
# X = list(map(tweet_to_words, df['clean_text']))

# from sklearn.preprocessing import LabelEncoder

# # Encode target labels
# le = LabelEncoder()
# Y = le.fit_transform(df['category'])

# y = pd.get_dummies(df['category'])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# from sklearn.feature_extraction.text import CountVectorizer
# #from sklearn.feature_extraction.text import TfidfVectorizer

# vocabulary_size = 5000

# # Tweets have already been preprocessed hence dummy function will be passed in
# # to preprocessor & tokenizer step
# count_vector = CountVectorizer(max_features=vocabulary_size,
# #                               ngram_range=(1,2),    # unigram and bigram
#                                 preprocessor=lambda x: x,
#                                tokenizer=lambda x: x)
# #tfidf_vector = TfidfVectorizer(lowercase=True, stop_words='english')

# # Fit the training data
# X_train = count_vector.fit_transform(X_train).toarray()

# # Transform testing data
# X_test = count_vector.transform(X_test).toarray()

# import sklearn.preprocessing as pr

# #Normalize BoW features in training and test set
# X_train = pr.normalize(X_train, axis=1)
# X_test  = pr.normalize(X_test, axis=1)

# max_words = 5000
# max_len=50

# def tokenize_pad_sequences(text):
#     '''
#     This function tokenize the input text into sequnences of intergers and then
#     pad each sequence to the same length
#     '''
#     # Text tokenization
#     tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
#     tokenizer.fit_on_texts(text)
#     # Transforms text to a sequence of integers
#     X = tokenizer.texts_to_sequences(text)
#     # Pad sequences to the same length
#     X = pad_sequences(X, padding='post', maxlen=max_len)
#     # return sequences
#     return X, tokenizer

# print('Before Tokenization & Padding \n', df['clean_text'][0])
# X, tokenizer = tokenize_pad_sequences(df['clean_text'])
# print('After Tokenization & Padding \n', X[0])

import pickle

# # saving
# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load model
model = load_model('best_model.h5')

def predict_class(text):
    '''Function to predict sentiment class of the passed text'''

    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len=50

    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = model.predict(xt).argmax(axis=1)
    # Print the predicted sentiment
    print('The predicted sentiment is', sentiment_classes[yt[0]])

test = []
text1 = ""
test.append(text1)
predict_class(test)


def tweet_to_words(tweet):
    ''' Convert tweet text into a sequence of words '''
    # convert to lowercase
    text = tweet.lower()
    # remove non letters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # tokenize
    words = text.split()
    # remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # apply stemming
    words = [PorterStemmer().stem(w) for w in words]
    # return list
    return words

test = tweet_to_words("Hello there @123")
print(test)

