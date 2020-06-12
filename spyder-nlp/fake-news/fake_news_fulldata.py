# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:15:08 2020

@author: kaustubh
"""

import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df_train = pd.read_csv("fake-news/train.csv")
df_test = pd.read_csv("fake-news/test.csv")

df_train.shape
df_train = df_train.dropna()
#df_test = df_test.dropna()

messages_train = df_train.copy()
messages_train.reset_index(inplace = True)


ps = PorterStemmer()
corpus = []
for i in range(0, len(messages_train)):
    review = re.sub('[^a-zA-z]', ' ', messages_train['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
corpus[389]
cv = CountVectorizer(max_features = 6000, ngram_range = (1, 3))
X = cv.fit_transform(corpus).toarray()
y = messages_train['label']

cv.get_feature_names()[:50]
count_df = pd.DataFrame(X, columns = cv.get_feature_names())


fake_news_model = MultinomialNB()

fake_news_model.fit(X, y)



messages_test = df_test.copy()
messages_test.reset_index(inplace = True)


ps = PorterStemmer()
corpus1 = []
for i in range(0, len(messages_test)):
    review = re.sub('[^a-zA-z]', ' ', str(messages_test['title'][i]))
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus1.append(review)


cv = CountVectorizer(max_features = 6000, ngram_range = (1, 3))
X_test = cv.fit_transform(corpus1).toarray()

test_pred = fake_news_model.predict(X_test)


output = pd.DataFrame({'id': df_test.id,'label': test_pred})
output.to_csv('submission.csv', index=False)
