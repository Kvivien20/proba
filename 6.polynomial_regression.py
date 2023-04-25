import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from feladat import generate_synthetic_data

def fit_polynomial_regression(x,y,degree):
    polynomial_regression=make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polynomial_regression.fit(x.reshape(-1,1),y)
    return polynomial_regression