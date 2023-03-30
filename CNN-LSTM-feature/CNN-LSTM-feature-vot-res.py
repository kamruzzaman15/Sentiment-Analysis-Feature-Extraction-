import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from tensorflow.keras.layers import Embedding
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Input
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM
from keras.models import Model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from keras.layers import Embedding, GlobalMaxPooling1D, BatchNormalization
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from math import sqrt
from sklearn.svm import SVC
import time
import tracemalloc
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
from codecarbon import EmissionsTracker

if __name__=='__main__':
    with EmissionsTracker() as tracker:
        st = time.time()
        tracemalloc.start()
        # Load dataset
        dataset = pd.read_csv('restaurant.csv')

        # Preprocessing
        texts = dataset['review'].values
        labels = dataset['label'].values

        # Tokenize the text and convert them into sequences of integers
        tokenizer = Tokenizer(num_words=8000)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        # Pad sequences so that they have equal lengths
        data = pad_sequences(sequences, maxlen=512)

        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.20, random_state=42)

        # Build the LSTM model
        inputs = Input(shape=(512,))
        x = Embedding(8000, 512, input_length=512)(inputs)
        x = Conv1D(filters=64, kernel_size=9, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x= Dropout(0.2)(x)
        x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(512, activation='relu', name='feature_layer')(x)
        out = Dense(1, activation='sigmoid')(x)
        full_model = Model(inputs=inputs, outputs=out)
        full_model.summary()


        full_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        full_model.fit(train_data, train_labels, epochs=5, batch_size=64)

        # Build feature extractor
        layer_name='feature_layer'
        feature_extractor = Model(inputs=full_model.input,
                                        outputs=full_model.get_layer(layer_name).output)

        feature_extractor.summary()

        # Get the extracted features from the last layer of the model

        train_features = feature_extractor.predict(train_data)
        test_features = feature_extractor.predict(test_data)


        # Fit a VotingClassifier to the extracted features and make predictions
        SVC_clf = SVC(probability=True, kernel='rbf')
        rf = RandomForestClassifier(random_state=1200, criterion='entropy', n_estimators=200)
        #ada = AdaBoostClassifier(n_estimators=200, learning_rate=2.0, algorithm='SAMME')
        ada = AdaBoostClassifier()

        #Voting Ensemble Method
        voting_clf = VotingClassifier(estimators=[('svm', SVC_clf), ('AdaBoost', ada), ('RF',rf )], voting='soft')
        voting_clf.fit(train_features, train_labels)
        preds = voting_clf.predict(test_features)

        acc = accuracy_score(test_labels, preds)
        rms= sqrt(mean_squared_error(test_labels, preds))


        print("Accuracy For Voting Ensemble is: " + str(acc))
        print("RMSE Error for Voting is: " + str(rms))

        et = time.time()
        elapsed_time = et - st
        print('Execution time for Voting is:', elapsed_time, 'seconds')
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print("Current memory usage for Voting is", current / (1024 * 1024), "MB; Peak was", peak / (1024 * 1024), "MB")

     
    print('Co2 for Voting:', tracker.final_emissions)
