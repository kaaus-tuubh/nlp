# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 23:21:16 2020

@author: kaustubh
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
 

messages  = pd.read_csv('SMSSpamCollectionn/SMSSPamCollection', sep = '\t', names = ["label", "message"])

#cleaning
corpus = []
ps = PorterStemmer()
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#BoW model
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()     #independent feature

y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values      #dependent feature


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

spam_detector_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detector_model.predict(X_test)

confusion_mat = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)