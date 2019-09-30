# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:54:35 2019

@author: TaiT_
"""

import pandas as pd

df = pd.read_csv('./data_sets/rating.csv')

# Since we are not considering time right now, drop it
df = df.drop(columns=['timestamp'])

# =============================================================================
# Since userIds are ordered sequentially from 1 to 138493
# And there are no missing numbers
# Subtract all userId by 1 to match df index
# =============================================================================

df.userId = df.userId - 1

# Since there are many repetitives movieId and the ids are missing
uniqueMovieIds = set(df.movieId.values)
movieIdToIndexHashmap = {}

movieIndex = 0
for movieId in uniqueMovieIds:
    movieIdToIndexHashmap[movieId] = movieIndex
    movieIndex += 1

# We will now use movie_idx instead of movieId
# Add new column to pandas dataframe, name it movie_idx
df['movie_idx'] = df.apply(lambda row: movieIdToIndexHashmap[row.movieId], axis=1)

df.to_csv('./data_sets/edited_rating.csv')
    
    



