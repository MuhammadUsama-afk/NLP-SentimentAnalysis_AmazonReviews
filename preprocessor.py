# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:33:44 2020

@author: Vegard
"""

import timeit
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import Counter

# Fetch stopwords from NTLK and stemming
word_stemmer = SnowballStemmer("english", ignore_stopwords=False)

# Creating a set of stopwords
stopWords = set([word for word in set(stopwords.words('english'))])


# Defining a function to clean up the text (removing new line symbols etc)
def cleanText(text):    
    text = text.replace('\n', ' ').strip().lower()
    # Remove any symbols
    text = re.sub(r'[^a-zæøåéäö ]+', '', text)
    # Remove consecutive whitespaces
    text = re.sub(r'\s\s+', ' ', text)
    # Removing stopwords in the text
    text = ' '.join([word_stemmer.stem(word) for word in text.split()
                     if word not in stopWords])
    return text


# Importing data start timer
print('Importing data')
start = timeit.default_timer()

# Set datafile
datafile =  'data/Seperated_Labels_test.csv'
# Load datafile
data = pd.read_csv(datafile)

# Removing unwanted variables
del datafile

# Importing data stop timer
stop = timeit.default_timer()
print('Time: ', stop - start)

# Cleaning up the text
# Start timer
print('Cleaning up the text')
start = timeit.default_timer()
data.Review = [cleanText(str(r)) if str(r) != 'nan' else '' for r in data.Review]
# Cleaning up the text stop timer
stop = timeit.default_timer()
print('Time: ', stop - start)

data.to_csv('data/amazon_clean_test.csv', index=False)