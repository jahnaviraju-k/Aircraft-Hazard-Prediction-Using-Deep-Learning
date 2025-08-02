from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns

main = tkinter.Tk()
main.title("Malware Detection")
main.geometry("1300x1200")

def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
    
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
    
def plotCorrelationMatrix(df, graphWidth):
    #filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        text.insert(END,f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.show()
def upload():
    global filename
    global data
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = ".")
    pathlabel.config(text=filename)
    text.insert(END,"Dataset loaded\n\n")
    data=pd.read_csv(filename)
    text.insert(END,"Data Information: \n"+str(data.head())+"\n")
    text.insert(END,"Columns Information: "+str(data.columns)+"\n")
    text.insert(END,"Shape of DataSet: "+str(data.shape)+"\n")
    data['classification'] = data.classification.map({'benign':0, 'malware':1})
    sns.countplot(data["classification"])
    plt.show()

def preprocess():
    global data
    global x,y
    text.delete('1.0',END)
    text.insert(END,"NA Values Information: \n"+str(data.isnull().sum())+"\n")
    data=data.dropna(how="any",axis=0)
    text.insert(END,"Output Variables Information:\n"+str(data["classification"].value_counts())+"\n")
    x=data.drop(["hash","classification",'vm_truncate_count','shared_vm','exec_vm','nvcsw','maj_flt','utime'],axis=1)
    y=data["classification"]

    plotPerColumnDistribution(data, 10, 5)

    plotCorrelationMatrix(data, 8)

    

def ttmodel():
    global x,y
    global X_train,X_test,y_train,y_test
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
    text.delete('1.0',END)
    text.insert(END,"Train set size :"+str(len(X_train))+"\n Test set size :"+str(len(X_test)))
def mlmodels():
    global clf_lr_acc,clf_svc_acc,clf_rfc_acc,clf_gnb_acc,clf_xgb_acc,clf_lgbm_acc
    clf_lr = LogisticRegression(random_state=0)
    clf_lr.fit(X_train,y_train)
    pred = clf_lr.predict(X_test)
    clf_lr_acc=clf_lr.score(X_test, y_test)
    text.insert(END,"Logistic Accuracy: "+str(clf_lr.score(X_test, y_test))+"\n")
    text.insert(END,"Logistic recall_score: "+str(recall_score(y_test,pred))+"\n")
    text.insert(END,"Logistic precision_score: "+str(precision_score(y_test,pred))+"\n")
    text.insert(END,"Logistic f1_score: "+str(f1_score(y_test,pred))+"\n")
    
    clf_svc = LinearSVC(random_state=0)
    clf_svc.fit(X_train,y_train)
    clf_svc.score(X_test,y_test)
    pred = clf_svc.predict(X_test)
    clf_svc_acc=clf_svc.score(X_test, y_test)
    text.insert(END,"SVC Accuracy: "+str(clf_svc.score(X_test, y_test))+"\n")
    text.insert(END,"SVC recall_score: "+str(recall_score(y_test,pred))+"\n")
    text.insert(END,"SVC precision_score: "+str(precision_score(y_test,pred))+"\n")
    text.insert(END,"SVC f1_score: "+str(f1_score(y_test,pred))+"\n")

    clf_gnb = GaussianNB()
    clf_gnb.fit(X_train,y_train)
    clf_gnb.score(X_test,y_test)
    pred = clf_gnb.predict(X_test)
    clf_gnb_acc=clf_gnb.score(X_test, y_test)
    text.insert(END,"GaussianNB Accuracy: "+str(clf_gnb.score(X_test, y_test))+"\n")
    text.insert(END,"GaussianNB recall_score: "+str(recall_score(y_test,pred))+"\n")
    text.insert(END,"GaussianNB precision_score: "+str(precision_score(y_test,pred))+"\n")
    text.insert(END,"GaussianNB f1_score: "+str(f1_score(y_test,pred))+"\n")

    clf_rfc = RandomForestClassifier(random_state=0)
    clf_rfc.fit(X_train,y_train)
    clf_rfc.score(X_test,y_test)
    pred = clf_rfc.predict(X_test)
    clf_rfc_acc=clf_rfc.score(X_test, y_test)
    text.insert(END,"RandomForest Accuracy: "+str(clf_rfc.score(X_test, y_test))+"\n")
    text.insert(END,"RandomForest recall_score: "+str(recall_score(y_test,pred))+"\n")
    text.insert(END,"RandomForest precision_score: "+str(precision_score(y_test,pred))+"\n")
    text.insert(END,"RandomForest f1_score: "+str(f1_score(y_test,pred))+"\n")
                
    clf_xgb = XGBClassifier()
    clf_xgb.fit(X_train, y_train)
    pred = clf_xgb.predict(X_test)
    clf_xgb_acc=clf_xgb.score(X_test, y_test)
    text.insert(END,"XGB Accuracy: "+str(clf_xgb.score(X_test, y_test))+"\n")
    text.insert(END,"XGB recall_score: "+str(recall_score(y_test,pred))+"\n")
    text.insert(END,"XGB precision_score: "+str(precision_score(y_test,pred))+"\n")
    text.insert(END,"XGB f1_score: "+str(f1_score(y_test,pred))+"\n")
    clf_lgbm = LGBMClassifier()
    clf_lgbm.fit(X_train, y_train)
    pred = clf_lgbm.predict(X_test)
    clf_lgbm_acc=clf_lgbm.score(X_test, y_test)
    text.insert(END,"LGBM Accuracy: "+str(clf_lgbm.score(X_test, y_test))+"\n")
    text.insert(END,"LGBM recall_score: "+str(recall_score(y_test,pred))+"\n")
    text.insert(END,"LGBM precision_score: "+str(precision_score(y_test,pred))+"\n")
    text.insert(END,"LGBM f1_score: "+str(f1_score(y_test,pred))+"\n")

def graph():
    global clf_lr_acc,clf_svc_acc,clf_rfc_acc,clf_gnb_acc,clf_xgb_acc,clf_lgbm_acc
    
    height = [clf_lr_acc,clf_svc_acc,clf_rfc_acc,clf_gnb_acc,clf_lgbm_acc,clf_lgbm_acc]
    bars = ('Logit', 'SVC','RFC','GNB','XGB','LGBM')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()   
                
font = ('times', 16, 'bold')
title = Label(main, text='A Malware Detection Method for Health Sensor Data Based on Machine Learning')
title.config(bg='dark salmon', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset and Import", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='dark orchid', fg='white')  
pathlabel.config(font=font1)
pathlabel.place(x=700,y=150)

dp = Button(main, text="Data Preprocessing", command=preprocess)
dp.place(x=700,y=200)
dp.config(font=font1)

tt = Button(main, text="Train and Test Model", command=ttmodel)
tt.place(x=700,y=250)
tt.config(font=font1)

ml = Button(main, text="Run Algorithms", command=mlmodels)
ml.place(x=700,y=300)
ml.config(font=font1)

gph = Button(main, text="Accuracy Graph", command=graph)
gph.place(x=700,y=350)
gph.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='old lace')
main.mainloop()



