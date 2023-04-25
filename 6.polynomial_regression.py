import numpy as np


def visualize_data_and_fit(x,y,model,degrees):
    plt.scatter(x,y)
    plt.xlabel("Feature(X)")
    plt.scatter("Target(y)")
    plt.title("Synthetic Data with Polynomial Relationship and Noise")
    x_pred=np.linspace(min(x), max(x), len(x)).reshape(-1,1)
    colors=['red','blue','green']
    for model, degree, color on zip(models, degrees, colors):
        y_pred=model.predict(x_pred)
        plt.plot(x_pred,y_pred,color=color,label=f'Polynomial Regression (degree {degree}')
    plt.legend()
    plt.show()