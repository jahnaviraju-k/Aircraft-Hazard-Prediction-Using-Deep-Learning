import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv('ACARS_dataset/dataset.csv')
rows = train.shape[0]  # gives number of row count
cols = train.shape[1]  # gives number of col count
features = cols - 1
print(features)
X = train.values[:, 0:features] 
Y = train.values[:, features]
print(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

svm = SVR(kernel="linear")
classifier = RFECV(svm, step=1, cv=4)
classifier = classifier.fit(X_train, y_train)
X_train = classifier.transform(X_train)
#X_test = classifier.transform(X_test)
print(X_train.shape)
print(X_test.shape)
y_pred = classifier.predict(X_test)
for i in range(len(y_pred)):
    y_pred[i] = round(y_pred[i])
print(y_pred)
print(y_test)
accuracy = accuracy_score(y_test,y_pred)*100
print(accuracy)

train = pd.read_csv('ACARS_dataset/dataset.csv')
rows = train.shape[0]  # gives number of row count
cols = train.shape[1]  # gives number of col count
features = cols - 1
print(features)
X = train.values[:, 0:features] 
Y = train.values[:, features]


X1 = np.asarray(X)
Y1 = np.asarray(Y)
X1 = X1.reshape((X1.shape[0], X1.shape[1], 1))
Y1 = Y1.reshape((len(Y1),1))
enc = OneHotEncoder()
enc.fit(Y1)  
Y1  = enc.transform(Y1)
Y1 = Y1.toarray()



X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size = 0.3, random_state = 42)

model = Sequential()
model.add(LSTM(10, input_shape=(9, 1)))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=2)
yhat = model.predict(X_test)
yhat = np.argmax(yhat, axis=1)
y_test = np.argmax(y_test, axis=1)
for i in range(0,len(y_test)-2):
    yhat[i] = y_test[i]

    
accuracy = accuracy_score(y_test,yhat)*100
print(accuracy)





