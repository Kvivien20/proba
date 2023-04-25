import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from feladat import generate_synthetic_data

def fit_polynomial_regression(x,y,degree):
    polynomial_regression=make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polynomial_regression.fit(x.reshape(-1,1),y)
    return polynomial_regression

def visualize_data_and_fit(x,y,model,degrees):
    plt.scatter(x,y)
    plt.xlabel("Feature(X)")
    plt.scatter("Target(y)")
    plt.title("Synthetic Data with Polynomial Relationship and Noise")
    x_pred=np.linspace(min(x), max(x), len(x)).reshape(-1,1)
    colors=['red','blue','green']
    for model, degree, color in zip(models, degrees, colors):
        y_pred=model.predict(x_pred)
        plt.plot(x_pred,y_pred,color=color,label=f'Polynomial Regression (degree {degree}')
    plt.legend()
    plt.show()