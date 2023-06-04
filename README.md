# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Assign the features for x and y respectively.
4. Split the x and y sets into train and test sets.
5. Convert the Alphabetical data to numeric using CountVectorizer.
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7. Find the accuracy of the model.


## Program:
```

Program to implement the SVM For Spam Mail Detection..

Developed by: Divyashree B S
RegisterNumber:  212221040044

print("Result Output:")
import chardet 
file='/content/spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows-1252')

print("data head:")
data.head()

print("data info:")
data.info()

print("data isnull:")
data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

print("y_prediction  value:")
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred

print("Accuracy Value:")
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

<img width="417" alt="exp9 op1" src="https://github.com/divvisha/Implementation-of-SVM-For-Spam-Mail-Detection/assets/127508123/2fbc0e74-567b-4826-aac3-2332ca8a77c2">

<img width="479" alt="exp9 op2" src="https://github.com/divvisha/Implementation-of-SVM-For-Spam-Mail-Detection/assets/127508123/bf634857-f03a-41c7-b180-06a96aebcb4f">

<img width="287" alt="exp9 op3" src="https://github.com/divvisha/Implementation-of-SVM-For-Spam-Mail-Detection/assets/127508123/1cb806db-572e-4fd3-9afd-3f6f4c410c09">

<img width="157" alt="exp9 op4" src="https://github.com/divvisha/Implementation-of-SVM-For-Spam-Mail-Detection/assets/127508123/e26c2a3c-0cf7-45c0-9c91-6e63cb4da144">

<img width="482" alt="exp9 op5" src="https://github.com/divvisha/Implementation-of-SVM-For-Spam-Mail-Detection/assets/127508123/ab48119f-5204-4162-8808-f34fcf7cb9ab">

<img width="285" alt="exp9 op6" src="https://github.com/divvisha/Implementation-of-SVM-For-Spam-Mail-Detection/assets/127508123/2eaf0e8f-bc09-4cca-baf0-03470c62d67a">


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
