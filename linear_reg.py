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

     print("--- Part 2: Multivariate Linear Regression (GasProperties.csv) ---")
    # (6) Multivariate Regression using Least Squares Method
    try:
# Load the dataset
        df = pd.read_csv("GasProperties.csv")
        
        # Split features (X) and target (Y - assume last column 'Idx')
        X_multi = df.iloc[:, :-1].values
        Y_multi = df.iloc[:, -1].values
        
        # Add a column of ones to X to account for the intercept term (w0)
        X_multi = np.c_[np.ones((X_multi.shape[0], 1)), X_multi]

        # Split into 80% training and 20% testing
        X_train, X_test, y_train, y_test = train_test_split(X_multi, Y_multi, test_size=0.2, random_state=42)

        # Calculate least squares weights: W = (X^T * X)^-1 * X^T * y
        W = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
        
        # Predict on training and testing sets
        y_train_pred = X_train.dot(W)
        y_test_pred = X_test.dot(W)
        
        # Compute RMSE
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        print("Multivariate Least Squares Results:")
        print(f"Weight vector W: {W}")
        print(f"Training RMSE: {rmse_train:.4f}")
        print(f"Testing RMSE: {rmse_test:.4f}")
        
    except FileNotFoundError:
        print("Error: GasProperties.csv not found. Ensure the file is in the same directory.")

if __name__ == '__main__':
    main()
