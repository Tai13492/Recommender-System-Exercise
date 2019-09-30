# -*- coding: utf-8 -*-
import pickle

import numpy as np
from sortedcontainers import SortedList

with open('./dictionaries/user2movie','rb') as file:
    user2movie = pickle.load(file)

with open('./dictionaries/movie2user','rb') as file:
    movie2user = pickle.load(file)

with open('./dictionaries/usermovie2rating','rb') as file:
    usermovie2rating = pickle.load(file)

with open('./dictionaries/usermovie2rating_test','rb') as file:
    usermovie2rating_test = pickle.load(file)

# =============================================================================
# a = ("MNNIT Allahabad", 5000, "Engineering")   
# this lines UNPACKS values 
# of variable a 
# (college, student, type_ofcollege) = a   
# =============================================================================

N = np.max(list(user2movie.keys())) + 1
print("Number of users in the training sets: " + str(N))

m1 = np.max(list(movie2user.keys()))
#temp = usermovie2rating_test.items()
#for (user,movie), rating in temp:
#    print(rating)
#temp2 = list(temp)[:1]
#for element in temp2:
#    print(type(element),'element')
#    print(type(element[0]))
#    print(element[0])
#    print(type(element[1]))
#    print(element[1])
m2 = np.max([movie for (user,movie), rating in usermovie2rating_test.items()])
M = max(m1,m2) + 1
print("Number of movies in the training sets: " + str(M))

if N > 10000:
    print("Exiting...., data sets too large")
    exit()

# No. of neighbors
K = 25 
# Minimum movies users have in common to be considered
limit = 5
neighbors = []
averages = []
deviations = []

for i in range(N):
    # List of movies that user i had rated
    movies_i = user2movie[i]
    movies_i_set = set(movies_i)
    
    # Dictionaries of ratings rated by user i
    ratings_i ={movie:usermovie2rating[(i,movie)] for movie in movies_i }
    # Average rating of user i
    avg_i = np.mean(list(ratings_i.values()))
    # Difference of each movies and user's i average rating
    dev_i = {movie:(rating - avg_i)for movie,rating in ratings_i.items()}
    dev_i_values = np.array(list(dev_i.values()))
    # Square the difference (x bar - x)
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))
    
    averages.append(avg_i)
    deviations.append(dev_i)
    
    sl = SortedList()
    for j in range(N):
        # Don't calculate yourself as your neightbor!
        if i == j:
            continue
        
        movies_j = user2movie[j]
        movies_j_set = set(movies_j)
        common_movies = movies_i_set.intersection(movies_j_set)
        
        if len(common_movies) < limit:
            continue
        
        ratings_j = {movie:usermovie2rating[(j,movie)] for movie in movies_j}
        avg_j = np.mean(list(ratings_j.values()))
        dev_j = {movie:(rating - avg_j) for movie,rating in ratings_j.items()}
        dev_j_values = np.array(list(dev_j.values()))
        sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))
        
        # Calculate correlation coefficient
        # Sigma [(x - xbar)(y - ybar)]
        numerator = sum(dev_i[movie_index] * dev_j[movie_index] for movie_index in common_movies)
        # Sqrt[Sigma(x - xbar)**2] * Sqrt[Sigma(y-ybar)**2]
        w_ij = numerator / (sigma_i*sigma_j)
        #print(w_ij)
        # Add in negative because Pearson corelation's formula and SortedList behaviour are different
        sl.add((-w_ij,j))
        # If find someone that is more corelate, drop someone who does not
        if len(sl) > K:
            del sl[-1]
    neighbors.append(sl)
    
    print(i," th loop")

def predict(i,movie):
    numerator = 0
    denominator = 0
    prediction = 0
    for neg_w, j in neighbors[i]:
        try:
            numerator += -neg_w * deviations[j][movie]
            denominator += abs(neg_w)
        except KeyError:
            pass
    if denominator == 0:
        prediction = averages[i]
    else:
        prediction = (numerator / denominator) + averages[i]
    prediction = min(5, prediction)
    prediction = max(0.5, prediction)
    return prediction

train_predictions = []
train_answers = []
for (i,movie), rating in usermovie2rating.items():
    prediction = predict(i,movie)
    train_predictions.append(prediction)
    train_answers.append(rating)

test_predictions = []
test_answers = []
for (i,movie), rating in usermovie2rating_test.items():
    prediction = predict(i,movie)
    test_predictions.append(prediction)
    test_answers.append(rating)

def mse(predictions,answers):
    p = np.array(predictions)
    a = np.array(answers)
    return np.mean((p-a)**2)

print("Training sets error: " + str(mse(train_predictions,train_answers)))
print("Testing sets error: " + str(mse(test_predictions,test_answers)))


    