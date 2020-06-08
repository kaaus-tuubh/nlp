# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:56:47 2020

@author: kaustubh
"""
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

paragraph = """Dr Avul Pakir Jainulabdeen Abdul Kalam was born on 15 October 1931, at Rameswaram in Tamil Nadu, India.
He specialized in Aeronautical Engineering from Madras Institute of Technology and became an eminent scientist. 
Dr Kalam was awarded the Padma Bhushan (1981), the Padma Vibhushan (1990), and India’s highest civilian award- 
the Bharat Ratna in 1997. In 2002, Dr Kalam became the 11th President of India. 
He was fondly called the people's President for being the first President in the 
country to connect with the youth via the internet.
Named as the 'Missile Man' of India for his contributions in the field, he redefined the Presidency during his tenure
from 2002 to 2007. He often spoke to children and the country's youth-- inspiring them to think big in life;
 he also penned a number of books. Dr Kalam succumbed to a massive cardiac arrest and breath his last on July 27,
 2015 while delivering a speech at IIM Shillong. Here are some of his evergreen quotes from his speeches. 
Dr Kalam's three step guide to achieve goals in life are: Finding an aim in life before you are twenty years old; 
Acquire knowledge continuously to reach this goal;
Work hard and persevere so you can defeat all the problems and succeed.
The challenge, my young friends, is that you have to fight the hardest battle,
 and ever stop fighting until you arrive at your destined place. 
What will be the tools with which you will fight this battle? They are: have a great aim in life,
continuously acquire the knowledge, work hard and persevere to realize the great achievement.

We are as young as our faith and as old as our doubts. We are also as young as our self-confidence and as old as our fears.
We are as young as our hopes and as old as our despairs. """
 
all_sentences = nltk.sent_tokenize(paragraph)
all_words = nltk.word_tokenize(paragraph)

#stemming 
stemmer = PorterStemmer()
sentences_stem = nltk.sent_tokenize(paragraph)
for i in range(len(sentences_stem)):
    words = nltk.word_tokenize(sentences_stem[i])
    words = [stemmer.stem(j) for j in words if j not in set(stopwords.words('english'))]
    sentences_stem[i] = ' '.join(words)
    
#lemmatization
lemmatizer = WordNetLemmatizer()
sentences_lemm = nltk.sent_tokenize(paragraph)
for i in range(len(sentences_lemm)):
    words = nltk.word_tokenize(sentences_lemm[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences_lemm[i] = ' '.join(words)

# cleaning the text
sentences = nltk.sent_tokenize(paragraph)
ps = PorterStemmer()
wordnet = WordNetLemmatizer()
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#Bag of words
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
    
#TfIDF     
cv = TfidfVectorizer()
Y = cv.fit_transform(corpus).toarray()
