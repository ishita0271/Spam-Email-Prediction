# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

"""Data Preprocessing"""

# load the dataset to pandas dataframe
raw_mail_data=pd.read_csv('spam.csv',encoding='latin-1')
# replace the null values with a null string
mail_data=raw_data.where((pd.notnull(raw_mail_data)),'')

mail_data=raw_data[['Category','Message']]

mail_data.head()

# label spam mail as 0 ; Non spam mail(ham) as 1
mail_data.loc[mail_data['Category'] == 'spam','Category', ] = 0
mail_data.loc[mail_data['Category'] == 'ham','Category', ] = 1

#separate the data as text and label .X -->text; Y -->label
X=mail_data['Message']
y=mail_data['Category']

print(X)
print('..........')
print(y)

#transform the text data to feature vectors that can be used as input to the svm model using TfidfVectorizer
#convert the text to lowercase letters

feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase='True')
X=feature_extraction.fit_transform(X)

#split the data as train and test data
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=100)

#convert y_train and y_test values to integers
y_train=y_train.astype('int')
y_test=y_test.astype('int')

"""Training the model --> **Support Vector Machine**"""

#training the SVM model with training data
model=LinearSVC()
model.fit(X_train,y_train)

"""Evaluation of the model"""

#prediction on training data
Predicted_y_train=model.predict(X_train)
accuracy_train=accuracy_score(y_train,Predicted_y_train)

print("Accuracy on training data: ",accuracy_train)

#prediction on test data
predicted_y_test=model.predict(X_test)
accuracy_test=accuracy_score(y_test,predicted_y_test)
print("Accuracy on test data:",accuracy_test)

input_mail= ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times.,,,"]

input_mail_features = feature_extraction.transform(input_mail)

prediction = model.predict(input_mail_features)
print(prediction)

if (prediction[0]==1):
  print("HAM MAIL")
else:
  print("HAM MAIL")
