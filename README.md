# DMML-CW2
Second Coursework Submission for Data Mining &amp; Machine Learning
A. Building an Ensemble
Ensemble learning uses multiple machine learning models to try to make better predictions on a dataset. 
An ensemble model works by training different models on a dataset and having each model make predictions individually. 
The predictions of these models are then combined in the ensemble model to make a final prediction.
In this task, you will be using a Voting Classifier in which the ensemble model makes the prediction by majority vote. 
For example, if we use three models and they predict [1, 0, 1] for the target variable, the final prediction that the ensemble model would 
make would be 1, since two out of the three models predicted 1.
You will use four different models to put into the Voting Classifier: k-Nearest Neighbors (KNN), Support Vector Machines (SVM), 
Random Forest (RF), and Logistic Regression (LR). Use the Scikit-learn library in Python to implement these methods and use the Facebook Metrics dataset.

B. Predicting the Price of NETFLIX Stock with LSTM Neural Networks
Build a Python program that can predict the price of a specific stock. This project is a great example of applying machine learning in finance.
As mentioned in the subtitle, we will be using the stock price history of Netflix, Inc. (NFXLX), a streaming company.
