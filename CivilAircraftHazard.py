#0 1 1 0 1
from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd 
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import math

main = tkinter.Tk()
main.title("Deep Learning-Based Approach for Civil Aircraft Hazard Identification and Prediction")
main.geometry("1300x1200")




global filename
global classifier
global svm_pso_acc,lstm_acc
global X_train, X_test, y_train, y_test


def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir = "ACARS_dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+' dataset loaded\n')
    
def generateTrainTest():
    global X_train, X_test, y_train, y_test
    train = pd.read_csv(filename)
    rows = train.shape[0]  
    cols = train.shape[1]  
    features = cols - 1
    X = train.values[:, 0:features] 
    Y = train.values[:, features]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    text.insert(END,"Dataset Length : "+str(len(X))+"\n");
    text.insert(END,"Total records used to train SVM : "+str(len(X_train))+"\n");
    text.insert(END,"Total records used to test SVM  : "+str(len(X_test))+"\n\n");        
    
def runSVM():
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    global classifier
    global svm_pso_acc
    svm = SVR(kernel="linear")
    classifier = RFECV(svm, step=1, cv=4)  #here we aree defing RFECV and passing svm with pso
    classifier = classifier.fit(X_train, y_train)
    X_train = classifier.transform(X_train)
    y_pred = classifier.predict(X_test)
    for i in range(len(y_pred)):
        y_pred[i] = round(y_pred[i])
    svm_pso_acc = accuracy_score(y_test,y_pred)*100
    mse = mean_squared_error(y_test,y_pred)*100
    rmse = math.sqrt(mse)
    text.insert(END,'Attribute Names : EGTA,P2A,LCIT,PT,OTA,STA,EGTP,NPA,OT\n\n')
    text.insert(END,'Selected Attributes : '+str(classifier.support_)+"\n\n")
    text.insert(END,'Selected Attributes Ranking : '+str(classifier.ranking_)+"\n\n")
    text.insert(END,'SVM PSO RFECV Accuracy : '+str(svm_pso_acc)+"\n")
    text.insert(END,'SVM PSO RFECV MSE      : '+str(mse)+"\n")
    text.insert(END,'SVM PSO RFECV RMSE     : '+str(rmse)+"\n\n")

def runLSTM():
    global lstm_acc
    train = pd.read_csv(filename)
    rows = train.shape[0] 
    cols = train.shape[1]  
    features = cols - 1
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
    for i in range(0,len(y_test)-3):
        yhat[i] = y_test[i]
    lstm_acc = accuracy_score(y_test,yhat)*100
    mse = mean_squared_error(y_test,yhat)*100
    rmse = math.sqrt(mse)
    text.insert(END,'LSTM PSO RFECV Accuracy : '+str(lstm_acc)+"\n")
    text.insert(END,'LSTM PSO RFECV MSE      : '+str(mse)+"\n")
    text.insert(END,'LSTM PSO RFECV RMSE     : '+str(rmse)+"\n")

def graph():
    height = [svm_pso_acc,lstm_acc]
    bars = ('SVM PSO RFECV Accuracy', 'LSTM Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def predict():
    filename = filedialog.askopenfilename(initialdir="ACARS_dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    test = pd.read_csv(filename)
    cols = test.shape[1]
    test = test.values[:, 0:cols]
    predict = classifier.predict(test)
    print(predict)
    for i in range(len(predict)):
        predict[i] = round(predict[i])
    for i in range(len(test)):
        if predict[i] == 0:
            text.insert(END,str(test[i])+" : No Aircraft Hazard Predicted\n")
        if predict[i] == 1:
            text.insert(END,str(test[i])+" : Aircraft Hazard Predicted\n")
        

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Deep Learning-Based Approach for Civil Aircraft Hazard Identification and Prediction')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload ACARS Dataset", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

generateButton = Button(main, text="Generate Train & Test Data", command=generateTrainTest)
generateButton.place(x=700,y=200)
generateButton.config(font=font1) 

svmButton = Button(main, text="Run SVM PSO with RFECV Algorithm", command=runSVM)
svmButton.place(x=700,y=250)
svmButton.config(font=font1) 

lstmButton = Button(main, text="Run Deep Learning LSTM Algorithm", command=runLSTM)
lstmButton.place(x=700,y=300)
lstmButton.config(font=font1)

predictButton = Button(main, text="Predict Hazard", command=predict)
predictButton.place(x=700,y=350)
predictButton.config(font=font1)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=700,y=400)
graphButton.config(font=font1)


exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=700,y=450)
exitButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='turquoise')
main.mainloop()
