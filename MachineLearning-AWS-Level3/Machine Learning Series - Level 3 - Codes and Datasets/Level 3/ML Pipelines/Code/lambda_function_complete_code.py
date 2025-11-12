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
data_s3_url = os.environ['data_s3_url']
model_bucket_name = os.environ['model_bucket_name']

# Defining a function that Imports the dataset
def read_csv_from_s3(s3_url):
    bucket_name = s3_url.split('/')[2].split('.')[0]
    file_key = '/'.join(s3_url.split('/')[3:])
    s3 = boto3.client('s3')
    csv_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    csv_string = csv_obj['Body'].read().decode('utf-8')
    csv_reader = csv.reader(StringIO(csv_string), delimiter=';')
    headers = next(csv_reader)
    data = [row for row in csv_reader]
    return headers, data

# Defining a function that Splits the dataset into the Training set and the Test set
def split_data(data, test_ratio=0.10):
    np.random.shuffle(data)
    split_index = int(len(data) * (1 - test_ratio))
    return data[:split_index], data[split_index:]

# Defining a function that Extracts the Features and Target from the Training set or Test set
def extract_features_and_target(headers, data):
    X = [list(map(float, row[:-1])) for row in data]
    y = [float(row[-1]) for row in data]
    return X, y

# Defining a function that Trains a Random Forest Model on the Training set, Evaluates it on the Test set, and Saves it in a specified S3 bucket
def train_evaluate_save(event, context):
    headers, raw_data = read_csv_from_s3(data_s3_url)
    train_data, test_data = split_data(raw_data)
    X_train, y_train = extract_features_and_target(headers, train_data)
    X_test, y_test = extract_features_and_target(headers, test_data)
    regressor = RandomForestRegressor(n_estimators=100)
    regressor.fit(X_train, y_train)
    model_name = 'Random-Forest-Model-' + str(random.randint(0, 100))
    temp_file_path = '/tmp/' + model_name
    with open(temp_file_path, 'wb') as f1:
        pickle.dump(regressor, f1)
    with open(temp_file_path, 'rb') as f2:
        model_data = f2.read()
    s3 = boto3.resource('s3')
    s3_object = s3.Object(model_bucket_name, model_name)
    s3_object.put(Body=model_data)
    y_predicted = regressor.predict(X_test)
    print("Mean Absolute Error: {:.3f}".format(metrics.mean_absolute_error(y_test, y_predicted)))
    return model_name

# Defining a function that Predicts the quality of a chosen wine using the pre-trained model saved in S3
def predict_wine_quality(event, context):
    input_data = [
        float(event['fixed acidity']),
        float(event['volatile acidity']),
        float(event['citric acid']),
        float(event['residual sugar']),
        float(event['chlorides']),
        float(event['free sulfur dioxide']),
        float(event['total sulfur dioxide']),
        float(event['density']),
        float(event['pH']),
        float(event['sulphates']),
        float(event['alcohol'])
    ]
    model_name = event['model name']
    temp_file_path = '/tmp/' + model_name
    s3 = boto3.client('s3')
    s3.download_file(model_bucket_name, model_name, temp_file_path)
    with open(temp_file_path, 'rb') as f:
        model = pickle.load(f)
    return str(round(model.predict([input_data])[0], 1))
