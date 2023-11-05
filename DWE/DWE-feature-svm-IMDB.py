import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import FastText
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import gensim.downloader as api
from nltk.tokenize import word_tokenize
import fasttext
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
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
        df = pd.read_csv("IMDB.csv")

        # Split the data into training and testing sets
        train_text, test_text, train_labels, test_labels = train_test_split(df["review"], df["label"], test_size=0.2, random_state=42)

        # Save the training data in the required format for FastText
        with open("train.txt", "w") as f:
            for text, label in zip(train_text, train_labels):
                f.write(f"__label__{label} {text}\n")
                
        # Train FastText on the training data
        model1 = fasttext.train_supervised(input="train.txt")

        # Use FastText to generate feature vectors for the training and testing data
        train_features_fasttext = np.array([model1.get_sentence_vector(text) for text in train_text])
        test_features_fasttext = np.array([model1.get_sentence_vector(text) for text in test_text])

        glove_model = api.load("glove-wiki-gigaword-300")


        train_features_glove = np.array([np.mean([glove_model.get_vector(word) for word in text if word.strip()], axis=0) for text in train_text])    
        test_features_glove = np.array([np.mean([glove_model.get_vector(word) for word in text if word.strip()], axis=0) for text in test_text])
        

        # Concatenate the feature vectors
        train_features = np.concatenate((train_features_fasttext, train_features_glove), axis=1)
        test_features = np.concatenate((test_features_fasttext, test_features_glove), axis=1)


        # Reset the classifiers
        clf = SVC()
        clf.fit(train_features, train_labels)
        preds = clf.predict(test_features)

        acc = accuracy_score(test_labels, preds)
        print("Accuracy For SVM IMDB is: " + str(acc))

        et = time.time()
        elapsed_time = et - st
        print('Execution time for SVM is:', elapsed_time, 'seconds')
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print("Current memory usage for SVM is", current / (1024 * 1024), "MB; Peak was", peak / (1024 * 1024), "MB")

     
    print('Co2 for SVM:', tracker.final_emissions)