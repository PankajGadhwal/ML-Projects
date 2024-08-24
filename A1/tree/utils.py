"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from scipy.stats import entropy as scipy_entropy

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """

    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded = encoder.fit_transform(X)
    
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(X.columns))
    
    return encoded_df

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """

    return np.issubdtype(y.dtype, np.floating)


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """

    if check_ifreal(Y):
        raise ValueError("Entropy is typically used for discrete values, not continuous.")
    
    label_encoder = LabelEncoder()
    encoded_Y = label_encoder.fit_transform(Y)
    
    value_counts = np.bincount(encoded_Y)
    return scipy_entropy(value_counts, base=2)


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """

    if check_ifreal(Y):
        raise ValueError("Gini index is typically used for discrete values, not continuous.")

    label_encoder = LabelEncoder()
    encoded_Y = label_encoder.fit_transform(Y)
    value_counts = np.bincount(encoded_Y)
    probabilities = value_counts / len(Y)
    return 1.0 - np.sum(probabilities ** 2)


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """

    if criterion == 'entropy':
        total_entropy = entropy(Y)
        weighted_entropy = 0.0
        for value in attr.unique():
            subset_Y = Y[attr == value]
            weighted_entropy += (len(subset_Y) / len(Y)) * entropy(subset_Y)
        return total_entropy - weighted_entropy

    elif criterion == 'gini':
        total_gini = gini_index(Y)
        weighted_gini = 0.0
        for value in attr.unique():
            subset_Y = Y[attr == value]
            weighted_gini += (len(subset_Y) / len(Y)) * gini_index(subset_Y)
        return total_gini - weighted_gini

    elif criterion == 'mse':
        total_mse = np.mean((Y - Y.mean()) ** 2)
        weighted_mse = 0.0
        for value in attr.unique():
            subset_Y = Y[attr == value]
            weighted_mse += (len(subset_Y) / len(Y)) * np.mean((subset_Y - subset_Y.mean()) ** 2)
        return total_mse - weighted_mse

    else:
        raise ValueError("Criterion must be 'entropy', 'gini', or 'mse'.")


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    best_gain = -float('inf')
    best_attribute = None
    best_split_value = None

    for feature in features:
        if check_ifreal(X[feature]):  # Regression
            sorted_values = sorted(X[feature].unique())
            for i in range(1, len(sorted_values)):
                split_value = (sorted_values[i - 1] + sorted_values[i]) / 2
                gain = information_gain(y, X[feature] <= split_value, criterion)
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = feature
                    best_split_value = split_value
        else:  # Classification
            for value in X[feature].unique():
                gain = information_gain(y, X[feature] == value, criterion)
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = feature
                    best_split_value = value

    return best_attribute, best_split_value


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    if check_ifreal(X[attribute]):
        X_left = X[X[attribute] <= value]
        y_left = y[X[attribute] <= value]
        X_right = X[X[attribute] > value]
        y_right = y[X[attribute] > value]
    else:
        X_left = X[X[attribute] == value]
        y_left = y[X[attribute] == value]
        X_right = X[X[attribute] != value]
        y_right = y[X[attribute] != value]
    
    return X_left, y_left, X_right, y_right
