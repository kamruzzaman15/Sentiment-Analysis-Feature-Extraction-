import pandas as pd
import numpy as np
import fasttext
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import time
import tracemalloc
import xgboost as xgb
from codecarbon import EmissionsTracker
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler


if __name__=='__main__':
    with EmissionsTracker() as tracker:
        st = time.time()
        tracemalloc.start()
        # Load the dataset
        df = pd.read_csv("restaurant.csv")

        # Split the data into training and testing sets
        train_text, test_text, train_labels, test_labels = train_test_split(df["review"], df["label"], test_size=0.2, random_state=42)

        # Save the training data in the required format for FastText
        with open("train.txt", "w") as f:
            for text, label in zip(train_text, train_labels):
                f.write(f"__label__{label} {text}\n")

        # Train FastText on the training data
        model = fasttext.train_supervised(input="train.txt")

        # Use FastText to generate feature vectors for the training and testing data
        train_features_fasttext = np.array([model.get_sentence_vector(text) for text in train_text])
        test_features_fasttext = np.array([model.get_sentence_vector(text) for text in test_text])

        # Use RoBERTa to generate feature vectors for the training and testing data
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModel.from_pretrained("roberta-base")
        train_features_roberta = np.zeros((len(train_text), 768))
        test_features_roberta = np.zeros((len(test_text), 768))
        for i, text in enumerate(train_text):
            input_ids = tokenizer.encode(text, max_length=512, return_tensors='pt')
            with torch.no_grad():
                #last_hidden_states = model(input_ids, output_hidden_states=True).hidden_states[11]
                last_hidden_states = model(input_ids).last_hidden_state
            train_features_roberta[i, :] = last_hidden_states[0, 0, :].numpy()
        for i, text in enumerate(test_text):
            input_ids = tokenizer.encode(text, max_length=512, return_tensors='pt')
            with torch.no_grad():
                #last_hidden_states = model(input_ids, output_hidden_states=True).hidden_states[11]
                last_hidden_states = model(input_ids).last_hidden_state
            test_features_roberta[i, :] = last_hidden_states[0, 0, :].numpy()

        # Concatenate the FastText and RoBERTa features
        train_features = np.concatenate((train_features_fasttext, train_features_roberta), axis=1)
        test_features = np.concatenate((test_features_fasttext, test_features_roberta), axis=1)

        # Scale the feature vectors
        scaler = MinMaxScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

        # Reset the classifiers
        svc = SVC(probability=True, kernel='rbf')
        nb = MultinomialNB()
        rf = RandomForestClassifier(random_state=1200, criterion='entropy', n_estimators=200)
        clf = VotingClassifier(estimators=[('svc', svc), ('nb', nb), ('rf', rf)], voting='soft')
        clf.fit(train_features, train_labels)
        preds = clf.predict(test_features)

        acc = accuracy_score(test_labels, preds)
        print("Accuracy For Voting restaurant is: " + str(acc))

        et = time.time()
        elapsed_time = et - st
        print('Execution time for Voting is:', elapsed_time, 'seconds')
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print("Current memory usage for Voting is", current / (1024 * 1024), "MB; Peak was", peak / (1024 * 1024), "MB")

     
    print('Co2 for Voting:', tracker.final_emissions)
