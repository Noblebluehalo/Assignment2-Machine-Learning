import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def main():
    print("--- Part 1: Simple 1D Linear Regression ---")
    # (1) Load training dataset
    X_simple = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 2.7]).reshape(-1, 1)
    Y_simple = np.array([15.5, 12.9, 9.5, 6.2, 5.8, 5.5, 7.1, 9.7, 13.5, 18.4, 21.4])

    # (2) Scatter plot using matplotlib
    plt.figure(figsize=(8, 6))
    plt.scatter(X_simple, Y_simple, color='blue', label='Data points')
    plt.title('Dataset Scatter Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
