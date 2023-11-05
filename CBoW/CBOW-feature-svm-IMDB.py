import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
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

        # Train the CBOW model
        model = Word2Vec(df['review'], window=15, min_count=1, sg=0)

        # Convert the text to numerical vectors using the CBOW model
        text_vectors = []
        for i in range(len(df)):
            text_vectors.append(np.mean([model.wv[word] for word in df['review'][i]], axis=0))
        text_vectors = np.array(text_vectors)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(text_vectors, df['label'], test_size=0.2, random_state=0)


        clf = SVC()
        clf.fit(X_train, y_train)

        # Predict the labels on the test set
        y_pred = clf.predict(X_test)

        # Evaluate the accuracy of the model
        acc = accuracy_score(y_test, y_pred)

        print("Accuracy For IMDB SVM is: " + str(acc))


        et = time.time()
        elapsed_time = et - st
        print('Execution time for SVM is:', elapsed_time, 'seconds')
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print("Current memory usage for SVM is", current / (1024 * 1024), "MB; Peak was", peak / (1024 * 1024), "MB")

     
    print('Co2 for SVM:', tracker.final_emissions)
