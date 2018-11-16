from sklearn import datasets
from scipy import sparse
import numpy as np

def get_data():
    
    digits = datasets.load_digits()
    X_train = digits.data
    y_train = digits.target

    return { "X" : X_train, "y" : y_train }
