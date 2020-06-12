# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:14:56 2020

@author: kaustubh
"""

import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv("C:\\Users\\kaustubh\\Desktop\\ld\\spyder-nlp\\tweet_fake\\train.csv")
df = df.drop('keyword', axis = 1)
df = df.drop('location', axis = 1)


df = df.dropna()

tweets = df.copy()
tweets.reset_index(inplace = True)

ps = PorterStemmer()
corpus = []
for i in range(0, len(tweets)):
    review = re.sub('[^a-zA-Z]', ' ', tweets['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


y = tweets['target']

cv = TfidfVectorizer(max_features = 5000, ngram_range = (1, 2))
X = cv.fit_transform(corpus).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

df_test = pd.read_csv("C:\\Users\\kaustubh\\Desktop\\ld\\spyder-nlp\\tweet_fake\\test.csv")

df_test = df_test.drop('keyword', axis = 1)
df_test = df_test.drop('location', axis = 1)


df_test = df_test.dropna()

corpus_test = []
ps = PorterStemmer()
for i in range(0, len(df_test)):
    review = re.sub('[^a-zA-Z]', ' ', df_test['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus_test.append(review)

cv = TfidfVectorizer(max_features = 5000, ngram_range = (1, 2))
y_testset = cv.fit_transform(corpus_test).toarray()

model.fit(X, y)

test_pred = model.predict(y_testset)


output = pd.DataFrame({'id': df_test.id,'target': test_pred})
output.to_csv('C:\\Users\\kaustubh\\Desktop\\ld\\spyder-nlp\\tweet_fake\\submission_tweet.csv', index=False)

