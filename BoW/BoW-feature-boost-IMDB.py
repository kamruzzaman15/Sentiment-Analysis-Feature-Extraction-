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
import xgboost as xgb
from codecarbon import EmissionsTracker

if __name__=='__main__':
    with EmissionsTracker() as tracker:
        st = time.time()
        tracemalloc.start()
        df=pd.read_csv('IMDB.csv')
        cv = CountVectorizer()
        cv.fit(df['review'])
        X = cv.transform(df['review'])

        train_text, test_text, train_labels, test_labels =train_test_split(X, df['label'], test_size=0.2,random_state=42)

        clf = xgb.XGBClassifier()

        #Voting Ensemble Method
        clf.fit(train_text, train_labels)
        preds = clf.predict(test_text)

        acc = accuracy_score(test_labels, preds)
        rms= sqrt(mean_squared_error(test_labels, preds))


        print("Accuracy For XBoost is: " + str(acc))
        print("RMSE Error for XBoost is: " + str(rms))

        et = time.time()
        elapsed_time = et - st
        print('Execution time for XBoost is:', elapsed_time, 'seconds')
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print("Current memory usage for XBoost is", current / (1024 * 1024), "MB; Peak was", peak / (1024 * 1024), "MB")

     
    print('Co2 for Voting:', tracker.final_emissions)