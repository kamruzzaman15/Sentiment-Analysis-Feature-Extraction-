import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import FastText
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from math import sqrt
from sklearn.svm import SVC
import time
import tracemalloc
import xgboost as xgb
from codecarbon import EmissionsTracker

if __name__=='__main__':
    with EmissionsTracker() as tracker:
        st = time.time()
        tracemalloc.start()
        # Load the dataset
        df = pd.read_csv('IMDB.csv')

        # Split the data into train and test sets
        train_data, test_data, train_labels, test_labels = train_test_split(df['review'], df['label'], test_size=0.2, random_state=42)

        # Load the FastText and Glove pre-trained models
        ft_model = api.load('fasttext-wiki-news-subwords-300')
        glove_model = api.load("glove-wiki-gigaword-300")

        # Define a function to extract hybrid features
        def hybrid_features(text):
            ft_feature = np.mean([ft_model.get_vector(word) for word in text if word.strip()], axis=0)
            glove_feature = np.mean([glove_model.get_vector(word) for word in text if word.strip()], axis=0)
            return np.concatenate((ft_feature, glove_feature))


        # Extract features for train and test sets
        train_features = np.array([hybrid_features(text) for text in train_data])
        test_features = np.array([hybrid_features(text) for text in test_data])

        clf = xgb.XGBClassifier()
        clf.fit(train_features, train_labels)

        # Make predictions on the test set
        predictions = clf.predict(test_features)

        # Evaluate the model using accuracy
        acc = accuracy_score(test_labels, predictions)
        rms= sqrt(mean_squared_error(test_labels, predictions))
        print('Accuracy for Xboost:', acc)
        print("RMSE Error for XBoost is: " + str(rms))

        et = time.time()
        elapsed_time = et - st
        print('Execution time for XBoost is:', elapsed_time, 'seconds')
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print("Current memory usage for XBoost is", current / (1024 * 1024), "MB; Peak was", peak / (1024 * 1024), "MB")

     
    print('Co2 for XBoost:', tracker.final_emissions)