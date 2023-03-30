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
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM, Bidirectional
from keras.models import Model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from keras.layers import Embedding, GlobalMaxPooling1D, BatchNormalization
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from math import sqrt
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
import time
import tracemalloc
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
from codecarbon import EmissionsTracker
from tensorflow.keras.optimizers import Nadam, SGD, Adagrad, Adadelta
import gensim.downloader as api

if __name__=='__main__':
    with EmissionsTracker() as tracker:
        st = time.time()
        tracemalloc.start()
        # Load dataset
        dataset = pd.read_csv('IMDB.csv')

        # Preprocessing
        texts = dataset['review'].values
        labels = dataset['label'].values

        # Tokenize the text and convert them into sequences of integers
        tokenizer = Tokenizer(num_words=30000)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        # Pad sequences so that they have equal lengths
        data = pad_sequences(sequences, maxlen=512)

        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.20, random_state=42)

        # Build the LSTM model
        inputs = Input(shape=(512,))
        x = Embedding(30000, 512, input_length=512)(inputs)
        x = Conv1D(filters=200, kernel_size=5, activation='relu')(x)
        x = MaxPooling1D()(x)
        x= Dropout(0.2)(x)
        x = Conv1D(filters=300, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D()(x)
        x= Dropout(0.2)(x)
        x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(x)
        x = Dense(512, activation='relu', name='feature_layer')(x)
        out = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=out)
        model.summary()

        model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_data, train_labels, epochs=10, batch_size=64)

        # Make predictions on the test data
        predictions = np.argmax(model.predict(test_data), axis=-1)

# Print the classification report and the accuracy
        print(classification_report(test_labels, predictions))
        score = model.evaluate(test_data, test_labels)
        print("Test accuracy for CNN-LSTM classifier %0.4f%%" % (score[1]*100))

        et = time.time()
        elapsed_time = et - st
        print('Execution time for CNN-LSTM is:', elapsed_time, 'seconds')
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print("Current memory usage for CNN-LSTM is", current / (1024 * 1024), "MB; Peak was", peak / (1024 * 1024), "MB")

     
    print('Co2 for CNN-LSTM:', tracker.final_emissions)
