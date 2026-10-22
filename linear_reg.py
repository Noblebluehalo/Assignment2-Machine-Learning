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

 # (3) Find the function using LinearRegression
    model = LinearRegression()
    model.fit(X_simple, Y_simple)
    
    # Plot the predicted line
    plt.plot(X_simple, model.predict(X_simple), color='red', label='Linear Fit')
    plt.legend()
    
    # Save the plot for the assignment report
    plt.savefig('scatter_plot.png')
    print("Saved scatter plot to 'scatter_plot.png'")

    # (4) Display coefficient vector and polynomial equation
    w1 = model.coef_[0]
    w0 = model.intercept_
    print(f"Coefficient vector: [w0, w1] = [{w0:.4f}, {w1:.4f}]")
    print(f"Equation: y = {w0:.4f} + {w1:.4f}*x1")

    # (5) Compute the training error (RMSE)
    y_pred = model.predict(X_simple)
    rmse_simple = np.sqrt(mean_squared_error(Y_simple, y_pred))
    print(f"RMSE (Simple Dataset): {rmse_simple:.4f}\n")
