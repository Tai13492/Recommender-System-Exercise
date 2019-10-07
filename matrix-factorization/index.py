# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:50:09 2019

@author: TaiT_
"""

import pickle
import numpy as np
import matplotlib as plt
from datetime import datetime

path_to_dir = '../user-user-collaborative-filtering/dictionaries'
with open(path_to_dir + '/user2movie','rb') as file:
    user2movie = pickle.load(file)

with open(path_to_dir + '/movie2user','rb') as file:
    movie2user = pickle.load(file)

with open(path_to_dir + '/usermovie2rating','rb') as file:
    usermovie2rating = pickle.load(file)

with open(path_to_dir + '/usermovie2rating_test','rb') as file:
    usermovie2rating_test = pickle.load(file)

# No. of users
N = np.max(list(user2movie.keys())) + 1

m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u,m), r in usermovie2rating_test.items()])
# No. of movies
M = max(m1,m2) + 1
print("N: ",N,"M: ",M)

# Initialize variables
K = 10 # Latent dimensionality
W = np.random.rand(N,K) # user's feature
b = np.zeros(N) # user's bias
U = np.random.rand(M,K) # movie's feature
c = np.zeros(M) # movie's bias
mu = np.mean(list(usermovie2rating.values())) # Global mean

def get_loss(l):
    # l is a list that contains um_dict dictionary
    # um_dict = (user_id,movie_id) : rating
    N = float(len(l))
    sse = 0
    for key, value in l.items():
        user_id, movie_id = key
        prediction = W[user_id].dot(U[movie_id]) + b[user_id] + c[movie_id] + mu
        sse += (prediction - value)**2
    return sse/N
        
# Training parameters
epochs = 25
reg = 0.01
train_losses = []
test_losses = []
for epoch in range(epochs):
    print("epoch: " + str(epoch))
    epoch_start = datetime.now()
    
    t0 = datetime.now()
    # User matrix
    for i in range(N):
        vector = np.zeros(K)
        matrix = np.eye(K) * reg
        user_bias = 0
        for j in user2movie[i]:
            r = usermovie2rating[(i,j)]
            matrix += np.outer(U[j],U[j])
            vector += (r-b[i] - c[j] - mu)*U[j]
            user_bias += (r - W[i].dot(U[j] - c[j] - mu))
        
        W[i] = np.linalg.solve(matrix, vector)
        b[i] = user_bias / (len(user2movie[i]) + reg)
        
        if i % (N//10) == 0:
            print("i: ",i, "N: ", N)
    print("updated W and b: ", datetime.now() - t0)
    
    t0 = datetime.now()
    # Movie matrix
    for j in range(M):
        vector = np.zeros(K)
        matrix = np.eye(K) * reg
        movie_bias = 0
        try:
            for i in movie2user[j]:
                r = usermovie2rating[(i,j)]
                matrix += np.outer(W[i],W[i])
                vector += (r - b[i] - c[j] - mu) * W[i]
                movie_bias += (r - W[i].dot(U[j]) - b[i] - mu)
            
            U[j] = np.linalg.solve(matrix,vector)
            c[j] = movie_bias / (len(movie2user[j]) + reg)
            
            if j % (M//10) == 0:
                print("j: ",j, "M: ",M)
        except KeyError:
            pass
    print("updated U and c: ", datetime.now() - t0)
    print("epoch duration: ", datetime.now() - epoch_start)
    
    t0 = datetime.now()
    train_losses.append(get_loss(usermovie2rating))
    
    test_losses.append(get_loss(usermovie2rating_test))
    
    print("calculate cost: ", datetime.now() - t0)
    print("train loss: ", train_losses[-1])
    print("test loss: ", test_losses[-1])
    

plt.plot(train_losses, label = "train loss")
plt.plot(test_losses, label = "test loss")
plt.legend()
plt.show()
            
        
    



