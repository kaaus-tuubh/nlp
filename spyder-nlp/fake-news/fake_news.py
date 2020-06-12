# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 23:16:55 2020

@author: kaustubh
"""


import pandas as pd
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df = pd.read_csv("fake-news/train.csv");

X = df.drop('label', axis=1)
y = df['label']

df.shape
df = df.dropna()

messages = df.copy()
messages.reset_index(inplace=True)

ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
cv = CountVectorizer(max_features = 5000, ngram_range = (1, 3))
X = cv.fit_transform(corpus).toarray()

y = messages['label']

cv.get_feature_names()[:20]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.30)

count_df = pd.DataFrame(X_train, columns = cv.get_feature_names())

fake_news_model = MultinomialNB()
fake_news_model.fit(X_train, y_train)

predict = fake_news_model.predict(X_test)

confusion_mat = confusion_matrix(y_test, predict)
accuracy = accuracy_score(y_test, predict)