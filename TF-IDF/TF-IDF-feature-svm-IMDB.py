import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_squared_error,recall_score,precision_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from math import sqrt
import time
import tracemalloc
from codecarbon import EmissionsTracker
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__=='__main__':
    with EmissionsTracker() as tracker:
        st = time.time()
        tracemalloc.start()
        
        df = pd.read_csv('IMDB.csv')
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(df['review'])


        clf = SVC()

        train_text, test_text, train_labels, test_labels =train_test_split(X, df['label'], test_size=0.2,random_state=42)

        clf.fit(train_text, train_labels)
        preds = clf.predict(test_text)

        acc = accuracy_score(test_labels, preds)
        rms= sqrt(mean_squared_error(test_labels, preds))


        print("Accuracy For SVM is: " + str(acc))
        print("RMSE Error for SVM is: " + str(rms))

        et = time.time()
        elapsed_time = et - st
        print('Execution time for SVM is:', elapsed_time, 'seconds')
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print("Current memory usage for SVM is", current / (1024 * 1024), "MB; Peak was", peak / (1024 * 1024), "MB")

     
    print('Co2 for SVM:', tracker.final_emissions)
