from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.optimizers import SGD

import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
from sklearn.metrics import accuracy_score, confusion_matrix

def normalize(data):
    featureName = list(data)
    for name in featureName:
        data[name] = (data[name]-np.min(data[name]))/(np.max(data[name])-np.min(data[name]))
    return data

data = pd.read_csv('data/trainComplete.csv')
X = data.ix[:, 'tempo':]
y = data['genre']

X = normalize(X)
y_categorical = np_utils.to_categorical(y)

# Creating a Neural Networks Model
model = Sequential()
model.add(Dense(29,input_shape=(X.shape[1],), activation='softmax'))
model.add(Dropout(0.2))
model.add(Dense(280, activation='softmax'))
model.add(Dropout(0.2))
model.add(Dense(350, activation='softmax'))
model.add(Dropout(0.2))
model.add(Dense(9, activation='softmax'))

# Compiling Neural Networks Model
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["acc"])

X_trainMatrix = np.array(X)
results = model.fit(X_trainMatrix, y_categorical, validation_split=0.3, epochs=1000, batch_size=32)

data2 = pd.read_csv('data/trainPartTest.csv')
x_test = data2.ix[:, 'tempo':]
y_test = data2['genre']

x_test = normalize(x_test)
y_categorical_test = np_utils.to_categorical(y_test)
X_trainMatrix_test = np.array(x_test)
scores = model.evaluate(X_trainMatrix_test, y_categorical_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

test_data = pd.read_csv('data/testComplete.csv')
X_test = test_data.ix[:,'tempo':]

X_test = normalize(X_test)
pred_X = model.predict(X_test.as_matrix())

test_pred = []
for row in pred_X:
    max_val, max_id = 0, 0
    for i in range(len(row)):
        if row[i] > max_val:
            max_id = i
            max_val = row[i]
    test_pred.append(max_id)

dataTrackId = open('data/test.csv', "r")
trackId= csv.reader(dataTrackId)
next(trackId)

result = open('data/results.csv', "wb")
writer = csv.writer(result)
writer.writerow(('track_id','genre_id'))

for row in test_pred:
    n = trackId.next()
    track_id = n[0]
    if (track_id == '098559'):
        writer.writerow((track_id,'4'))
        n = trackId.next()
        track_id = n[0]
    elif (track_id == '098571'):
        writer.writerow((track_id,'2'))
        n = trackId.next()
        track_id = n[0]
    writer.writerow((track_id,row))

dataTrackId.close()
result.close()
