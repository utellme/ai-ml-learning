# Machine Learning Level 3: Build an Automated Machine Learning Pipeline on AWS Lambda

# Importing the libraries
import os
import csv
import boto3
import pickle
import random
import logging
import numpy as np
from io import StringIO
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

# Defining the Environment Variables


# Defining a function that Imports the dataset


# Defining a function that Splits the dataset into the Training set and the Test set


# Defining a function that Extracts the Features and Target from the Training set or Test set


# Defining a function that Trains a Random Forest Model on the Training set, Evaluates it on the Test set, and Saves it in a specified S3 bucket


# Defining a function that Predicts the quality of a chosen wine using the pre-trained model saved in S3
