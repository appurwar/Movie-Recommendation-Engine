#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 22:43:47 2018

Movie Recommender System using Matrix Factorization
@author: apoorv
"""


import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import random


# String Literals 
userId = 'userId'
movieId = 'movieId'
rating = 'rating'

# Function to transform given data into dictionary
def getTrainAndTestData(ratings):
    
    ratings_dict_train = dict()
    ratings_dict_test = dict()
    movie_id_dict = dict()
    movie_rating_list = list()
    
    movie_count = 0
    for movie in sorted(ratings[movieId].unique()):
        movie_id_dict[movie] = movie_count
        movie_count += 1
        
    userID = 0
    # Iterate over all users and create a dictionary - {userID : {movieID : rating}}
    for index, row in ratings.iterrows():
        if(int(row[userId]) == userID):
            movie_rating_list.append({movie_id_dict[row[movieId]]:row[rating]})
        else:
            data = np.array(movie_rating_list)
            x_train_50, x_test_50 = train_test_split(data,test_size=0.5) 
            ratings_dict_train[userID] = x_train_50
            ratings_dict_test[userID] = x_test_50
            # Empty cuurent list - add next user's records
            movie_rating_list = list()
            movie_rating_list.append({movie_id_dict[row[movieId]]: row[rating]})
            userID += 1
    
    # Handles the data for last user
    data = np.array(movie_rating_list)
    x_train_50, x_test_50 = train_test_split(data,test_size=0.5)
    ratings_dict_train[userID] = x_train_50
    ratings_dict_test[userID] = x_test_50  
    
    # Write data to file to avoid recomputation later
    """
    output = open('train_data_50_new.pkl', 'wb')
    pickle.dump(ratings_dict_train, output, protocol=pickle.HIGHEST_PROTOCOL)
    output = open('test_data_50_new.pkl', 'wb')
    pickle.dump(ratings_dict_test, output, protocol=pickle.HIGHEST_PROTOCOL)
    """
    
    return ratings_dict_train, ratings_dict_test, movie_count

# Function to compute Matrix Factorization
def calcMatrixFactorization(training_matrix, movie_count, mean_rating, rank, alpha, beta, iters):
    
    user_count = len(ratings_dict_train)
    
    # Randomly initialize the thin matrices
    V = np.random.rand(user_count, rank)
    W = np.random.rand(movie_count, rank)
   
    # Initialize the biases
    b_user = np.random.uniform(-1,1,user_count)
    b_movie = np.random.uniform(-1,1,movie_count)
    
    #b_user = np.random.uniform(-0.05,0.05,user_count)
    #b_movie = np.random.uniform(-0.05,0.05,movie_count)
    
    # Global Bias
    b_global = mean_rating
    
    
    # The loops below assures iterations to be - "iters * No. of entries"
    for i in range(0,iters):   # Number of epochs
        print(i)
        # Since dictionary is unordered, our data here is effectively shuffled again here
        for user, value in sorted(training_matrix.items(), key = lambda x:random.random()): # Iterate for all users
            for data in value:
                for movie, rate in data.items():    # Iterate for all movies rated by this user
                    
                    # Prediction with current weights
                    prediction = V[user, :].dot(W[movie, :].T) + b_global + b_user[user] + b_movie[movie]
                    
                    # Compute the error for current prediction
                    error = rate - prediction 
                    #print(error)
                    # Update bias parameters
                    b_user[user] += alpha * (error - beta * b_user[user])
                    b_movie[movie] += alpha * (error - beta * b_movie[movie])
                    
                    
                    # Update feature matrix
                    V[user, :] += alpha * (error * W[movie, :] - beta * V[user,:])
                    W[movie, :] += alpha * (error * V[user, :] - beta * W[movie,:])
                    #print(userId, movieId, ratings)
            if(user%10000 == 0):
                print(user)
    
    return V, W, b_user, b_movie 


# Function to calculate the root mean square error
def calcRMSE(test_matrix, V, W, test_entry_count, b_user, b_movie, b_global):
    
    rmse = 0
    for user, value in test_matrix.items(): # Iterate for all users
        for data in value:
            for movie, rate in data.items():    # Iterate for all movies rated by this user
                
                rmse += pow(rate - (V[user].dot(W[movie].T)) - b_global - b_user[user] - b_movie[movie], 2) 
            
    rmse = rmse/test_entry_count
    
    return np.sqrt(rmse)                

# Function to calculate the Mean Recipocal Rank
def calcMRR(test_matrix, V, W, b_global, b_user, b_movie):
    
     curr_user = 1
     mrr = []
     list_rating = list()
     list_predicition = list()
     
     # Iterate over all users and movies
     for user, value in test_matrix.items():
         for data in value:
             for movie, rate in data.items():
                 if user == curr_user and rate >= 3:
                     # Store all movies with rating more than 3 for current user
                     list_rating.append([movie, rate])
                 if user == curr_user:
                     # Store predictions for all movies for current user
                     prediction  = prediction = V[user-1, :].dot(W[movie, :].T) + b_global + b_user[user-1] + b_movie[movie]
                     list_predicition.append([movie, prediction])
                 else:
                    # Sort the movie predictions in decreasing order
                    list_predicition = sorted(list_predicition, key = lambda l:l[1], reverse = True)
                    rank = 0
                    movie_list = list()
                    # Get movie ids
                    for pred in list_predicition:
                        movie_list.append(pred[0])
                    # Get index and rank 
                    for rating in range(len(list_rating)):
                        index, = np.where(np.array(movie_list) == np.array(list_rating[rating][0]))
                        rank += 1/(index+1)
                    # For all ranked movies, store the average rating for the user
                    if rank > 0:
                        mrr.append(rank/len(list_rating))
                    else:
                        # Since the movie is not rated
                        mrr.append(0)
                        
                    # Compute for the last user, as curr_user will not handle that case
                    list_rating = list()
                    list_predicition = list()
                    list_rating.append([movie, rate])
                    prediction  = prediction = V[user-1, :].dot(W[movie, :].T) + b_global + b_user[user-1] + b_movie[movie]
                    list_predicition.append([movie, prediction])
                    curr_user = user
    
     list_predicition = sorted(list_predicition, key = lambda l:l[1], reverse = True)
     rank  = 0
     movie_list = list()
     for pred in list_predicition:
         movie_list.append(pred[0])
     # Get index and rank 
     for rating in range(len(list_rating)):
         index, = np.where(np.array(movie_list) == np.array(list_rating[rating][0]))
         rank += 1/(index+1)
     # For all ranked movies, store the average rating for the user
     if rank > 0:
         mrr.append(rank/len(list_rating))
     else:
         # Since the movie is not rated
         mrr.append(0)
  
     return np.mean(mrr)
                        

if __name__ == "__main__":
    
    # Open the data file
    ratings = pd.read_csv('ratings.csv')
    
    entry_count = len(ratings['userId'])
    
    # Mean rating of all the movies
    mean_rating = sum(ratings['rating'])/len(ratings['rating'])
    
    # Call to the function to get the data in the form of dictionary
    #ratings_dict_train, ratings_dict_test, movie_count = getTrainAndTestData(ratings)
    
    # Code to read from pickle file
    
    print("Loading Training Data..")
    pkl_file = open('train_data_50_new.pkl', 'rb')
    ratings_dict_train = pickle.load(pkl_file)
    
    
    print("Loading Test Data..")
    pkl_file = open('test_data_50_new.pkl', 'rb')
    ratings_dict_test = pickle.load(pkl_file)
    
    pkl_file.close()
    
    #Comment following when processing data
    movie_count = 26744
    # Beta values for which we will test our function
    beta_values = [0.2, 0.02, 0.002, 0.0002]
    
    # Get results for multiple values of rank and beta
    for rank in range(4, 16, 4):
        for beta in beta_values:
            # Call to matrix factorization function
            V, W, b_user, b_movie = calcMatrixFactorization(ratings_dict_train, movie_count, mean_rating, rank, 0.01, beta, 4)
            
            # Since it is a 50-50 split
            test_entry_count = int(entry_count*0.5)
            
            # Calculate TRAIN RMSE
            rmse = calcRMSE(ratings_dict_train, V, W, test_entry_count, b_user, b_movie, mean_rating)
            rmse_train_result = []
            rmse_train_result.append([rank, beta, rmse])
            
            print("Train RMSE -")
            print(rank, beta, rmse)
            
            # Calculate TEST RMSE
            rmse = calcRMSE(ratings_dict_test, V, W, test_entry_count, b_user, b_movie, mean_rating)
            rmse_test_result = []
            rmse_test_result.append([rank, beta, rmse])
            
            print("Test RMSE -")
            print(rank, beta, rmse)
            
            # Training MRR
            mrr = calcMRR(ratings_dict_train, V, W, mean_rating, b_user, b_movie)
            print("Train MRR - ")
            print(rank, beta, mrr)
            mrr_train_result = []
            mrr_train_result.append([rank, beta, rmse])
            
            
            # Test MRR
            mrr = calcMRR(ratings_dict_test, V, W, mean_rating, b_user, b_movie)
            print("Test MRR - ")
            print(rank, beta, mrr)
            mrr_test_result = []
            mrr_test_result.append([rank, beta, rmse])
            
    
    rmse_train_result = np.array(rmse_train_result)
    np.save("rmse_train_result.npz", rmse_train_result)
    rmse_test_result = np.array(rmse_test_result)        
    np.save("rmse_test_result.npz", rmse_test_result) 
    mrr_train_result = np.array(mrr_train_result)
    np.save("mrr_train_result.npz", mrr_train_result)
    mrr_test_result = np.array(mrr_test_result)
    np.save("mrr_test_result.npz", mrr_test_result)
            
    
    



    
    
    