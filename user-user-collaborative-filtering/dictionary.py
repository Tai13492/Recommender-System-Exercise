# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 12:21:40 2019

@author: TaiT_
"""

import pandas as pd
from sklearn.utils import shuffle
import pickle

df = pd.read_csv('./data_sets/small_rating.csv')

N = df.userId.max() + 1 # No. of users
M = df.movie_idx.max() + 1 # No. of movies

print("Shuffling data")
# =============================================================================
# l = [1, 2, 3, 4, 5]
# head = l[0]
# tail = l[1:]
# =============================================================================
# Seperate data in to train and test
df = shuffle(df)
cutoff = int(0.8 * len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]


# Useful dictionaries
user2movie = {}
movie2user = {}
usermovie2rating = {}

count = 0
def constructDictionary(row):
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: " + str(float(count)/len(df_train)))
    
    i = int(row.userId)
    j = int(row.movie_idx)
    
    if i not in user2movie:
        user2movie[i] = [j]
    else:
        user2movie[i].append(j)
    
    if j not in movie2user:
        movie2user[j] = [i]
    else:
        movie2user[j].append(i)
        
    usermovie2rating[(i,j)] = row.rating

df_train.apply(constructDictionary, axis=1)

count = 0
usermovie2rating_test = {}
def constructTestDictionary(row):
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: " + str(float(count)/len(df_test)))
    
    i = int(row.userId)
    j = int(row.movie_idx)
    
    usermovie2rating_test[(i,j)] = row.rating

df_test.apply(constructTestDictionary, axis=1)

print("Dumping py objects")

with open('user2movie','wb') as file:
    pickle.dump(user2movie, file)

with open('movie2user','wb') as file:
    pickle.dump(movie2user, file)

with open('usermovie2rating','wb') as file:
    pickle.dump(usermovie2rating, file)

with open('usermovie2rating_test','wb') as file:
    pickle.dump(usermovie2rating_test, file)

print("Success")


    