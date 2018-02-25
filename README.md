# Movie Recommendation Engine - 

**Overview and Approach** – 

The task of creating a movie recommender system using matrix factorization on MovieLens database is completed using Python 3 as the programming language and all functions were executed on Google cloud.  The task is subdivided into 4 main parts – 

1)	**Processing the data** - _getTrainAndTestData()_: 
This method fetches the data from ‘ratings.csv’, re-indexing the ‘userIds’ and ‘movieIds’ starting from index 1, and removing any gaps in the IDs, thereby making it continuous and helping us save a lot of memory.  Then it converts this data into a dictionary of dictionaries of the form – {userID: {movieID, rating}} and then partitions the data in a manner that we have an equal number of movies for each user in both training and test data and save these entries in a file for future reuse. This is a one-time activity saves us a lot of time and memory when performing further computation on this huge dataset, making processing efficient. 

2)	**Matrix Factorization** – _calcMatrixFactorization()_: 
This function performs the matrix factorization step, creating 2 new skinny matrix of dimensions (users * rank) and (movies * rank).  These matrices are first initialized randomly and then based on the prediction of the movie rating and the amount which they are off from the actual results, stochastic gradient descent (SGD) is used to populate the values in these matrices.  Moreover, a global bias term and one bias each for users and helps us in fine tuning our results further. The addition of bias term helped me in significantly reducing the RMSE. While without bias the RMSE value was around 2, adding bias parameters reduced it further to around 0.9, which was further improved by parameter tuning, as discussed in the results section below. 

3)	**RMSE Calculation **– _calcRMSE()_: 
This method calculates and returns the root mean squared error of given dataset, using the skinny feature matrices generated using the matrix factorization method stated above. 

4)	**MRR Calculation** – _calcMRR()_: 
This method calculates and returns the Mean Reciprocal Rank of given dataset using the skinny feature matrices generated using the matrix factorization method. 
  Here, I have evaluated this recommender system for ranks ranging from “4 to 16” in a step size of 4, and beta values – “0.2, 0.02, 0.002 and 0.0002”. A logarithmic scale tends to expose the most variations for beta values, hence the choice of these values.
 
The results for the various combination of these values for 4 epochs ea , are presented in the next section along with graphs. The step sized for SGD used for all iterations was of 0.01.

I executed the code for a total of 3 times for all the values of rank and lambda, in order to gauge the mean and standard deviation in the predictions. The train and test RMSE and MRR displayed below are from the first run.

All the results are stored as numpy arrays to revisit the results in future.
 
# Results – 

## Surface Plots – 

•	**For Train RMSE** –  

![](https://github.com/appurwar/Movie-Recommendation-Engine/blob/master/SurfacePlotRMSE_Train.png)

Figure: Surface Plot for Train RMSE (Rank vs Lambda vs RMSE)



•	**For Test RMSE** – 

![](https://github.com/appurwar/Movie-Recommendation-Engine/blob/master/SurfacePlotRMSE.png)

Figure: Surface Plot for Test RMSE (Rank vs Lambda vs RMSE) 

•	**For Train MRR** – 

![](https://github.com/appurwar/Movie-Recommendation-Engine/blob/master/SurfacePlotMRR_Train.png)
  
Figure: Surface Plot for Train MRR (Rank vs Lambda vs MRR)


•	**For Test MRR** – 

![](https://github.com/appurwar/Movie-Recommendation-Engine/blob/master/SurfacePlotMRR.png)

Figure: Surface Plot for Test MRR (Rank vs Lambda vs MRR)

## Mean and Standard Deviation –

•	**For RMSE** (over 3 iterations) – 
![](https://github.com/appurwar/Movie-Recommendation-Engine/blob/master/RMSE_Mean_SD.png)

Figure 9: Mean and Standard Deviation (X-Axis: Rank, Y-Axis: RMSE)

•	**For MRR** (over 3 iteration) –  
![](https://github.com/appurwar/Movie-Recommendation-Engine/blob/master/MRR_Mean_SD.png)

Figure 10: Mean and Standard Deviation (X-Axis: Rank, Y-Axis: MRR)

# **Conclusion** –

As observed from the results, RMSE decreases and MRR increases as we increase the rank from 4 to 16 for training error, but after rank 12 the model shows signs of slight overfitting, visible from the difference in the training and test errors. 

Similarly, for lambdas RMSE decreases and MRR increases when we decrease the value of lambda from 0.2 to 0.0002.
Also, as the number of epochs increases on the data, RMSE decreases further and MRR increases, but the increase is minor after the initial few epochs and the improvement seems to plateau.
