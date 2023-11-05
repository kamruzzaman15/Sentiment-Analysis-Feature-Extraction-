import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch
import time
import tracemalloc
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from math import sqrt
from sklearn.svm import SVC
import time
import tracemalloc
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from codecarbon import EmissionsTracker
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

if __name__=='__main__':
    with EmissionsTracker() as tracker:
        st = time.time()
        tracemalloc.start()
        # Load the dataset
        df = pd.read_csv('product.csv')
        scaler = MinMaxScaler()

        # Prepare the inputs and labels
        texts = df['review'].values
        labels = df['positive_review'].values

        # Load the RoBERTa tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModel.from_pretrained("roberta-base")

        # Extract features using RoBERTa
        inputs = np.zeros((len(texts), 768))
        for i, text in enumerate(texts):
            input_ids = tokenizer.encode(text, return_tensors='pt')
            with torch.no_grad():
                last_hidden_states = model(input_ids, output_hidden_states=True).hidden_states[11]
            inputs[i, :] = last_hidden_states[0, 0, :].numpy()

        # Split the data into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)
        
        X_train_val = scaler.fit_transform(X_train_val)
        X_test = scaler.fit_transform(X_test)
            
        # Reset the classifiers
        svc = SVC(probability=True, kernel='rbf')
        nb = MultinomialNB()
        rf = RandomForestClassifier(random_state=1200, criterion='entropy', n_estimators=200)
        voting_clf = VotingClassifier(estimators=[('svc', svc), ('nb', nb), ('rf', rf)], voting='soft')

        # Evaluate the voting ensemble classifier on the test set
        voting_clf.fit(X_train_val, y_train_val)
        test_pred = voting_clf.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        print("Test accuracy for Voting product:", test_acc)

        et = time.time()
        elapsed_time = et - st
        print('Execution time for Voting is:', elapsed_time, 'seconds')
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print("Current memory usage for Voting is", current / (1024 * 1024), "MB; Peak was", peak / (1024 * 1024), "MB")

     
    print('Co2 for Voting:', tracker.final_emissions)
