# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 10:15:58 2019

@author: TaiT_
"""

# =============================================================================
# Select users who rated the most movies
# Movies who've been rated by the most users
# =============================================================================

import pandas as pd
from collections import Counter

df = pd.read_csv('./data_sets/edited_rating.csv')
print("Original dataframe size: ", len(df))

N = df.userId.max() + 1 # No. of users
M = df.movie_idx.max() + 1 # No. of movies

user_ids_count = Counter(df.userId)
movie_index_count = Counter(df.movie_idx)

# print(type(user_ids_count),'type of user_ids_count')

n = 10000 # No. of users we would like to keep
m = 2000 # No. of movies we would like to keep

# =============================================================================
# for user_id, count in user_ids_count.most_common(n):
#     print(user_id,'this is user_id')
#     print(count,'this is count')
# =============================================================================

user_ids = [user_id for user_id, count in user_ids_count.most_common(n)]
movie_indexes = [movie_index for movie_index, count in movie_index_count.most_common(m)]

df_small = df[df.userId.isin(user_ids) & df.movie_idx.isin(movie_indexes)].copy()

print(df_small.head(5))

print("Mapping userId and movie_index")
# Change movie_idx and userId again to make it sequential
new_user_id_map = {}
user_count = 0
for old_user_id in user_ids:
    new_user_id_map[old_user_id] = user_count
    user_count += 1

new_movie_index_map = {}
movie_count = 0
for old_movie_index in movie_indexes:
    new_movie_index_map[old_movie_index] = movie_count
    movie_count += 1

print("Setting new userId... ")
df_small.loc[:,'userId'] = df_small.apply(lambda row: new_user_id_map[row.userId],axis = 1)
print("Setting new movie_idx...")
df_small.loc[:,'movie_idx'] = df_small.apply(lambda row: new_movie_index_map[row.movie_idx], axis = 1 )

print(df_small.head(5))
print("Dataframe size: ", len(df_small))

df_small.to_csv("./data_sets/small_rating.csv")



