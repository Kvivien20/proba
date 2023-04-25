import numpy as np
import matplotlib.pyplot as plt
from feladat import LinearRegresssion

def fit_linear_regression(x,y):
    lr= LinearRegression()
    lr.fit(x.reshape(-1,1),y)
    return lr

def visualize_data_and_fit(x,y,model):
    plt.scatter(x,y)
    plt.xlabel("Feature(X)")
    plt.scatter("Target(y)")
    plt.title("Synthetic Data with Polynomial Relationship and Noise")
    x_pred=np.linspace(min(x), max(x), len(x)).reshape(-1,1)
    y_pred=model.predict(x_pred)
    plt.plot(x_pred, y_pred, color='red', label='Linear Regression Fit')
    plt.legend()
    plt.show()
