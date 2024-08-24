import os
import pandas as pd
import numpy as np

base_dir = 'Combined'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
headers = None

# Getting the train data
X_train = []
y_train = []

for category in os.listdir(train_dir):
    category_path = os.path.join(train_dir, category)
    if os.path.isdir(category_path):
        for file_name in os.listdir(category_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(category_path, file_name)
                if headers is None:
                    headers = pd.read_csv(file_path, nrows=1).columns.tolist()
                df = pd.read_csv(file_path)
                X_train.append(df.values)
                y_train.extend([category] * len(df))

X_train = np.vstack(X_train)
y_train = np.array(y_train)

# Getting the test data
X_test = []
y_test = []

for category in os.listdir(test_dir):
    category_path = os.path.join(test_dir, category)
    if os.path.isdir(category_path):
        for file_name in os.listdir(category_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(category_path, file_name)
                if headers is None:
                    headers = pd.read_csv(file_path, nrows=1).columns.tolist()
                df = pd.read_csv(file_path)
                X_test.append(df.values) 
                y_test.extend([category] * len(df)) 

X_test = np.vstack(X_test)
y_test = np.array(y_test) 

print("Headers: ", headers)
print("Training data shape: ", X_train.shape)
print("Testing data shape: ", X_test.shape) 
print("Training labels shape: ", y_train.shape)
print("Testing labels shape: ", y_test.shape)
