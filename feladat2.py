
def main():
    coefficients=[100,0.02,-0.002,0.014]
    x_values = np.linspace(-10,10,100)
    x,y=generate_synthetic_data(x_values,coefficients)
    lr=fit_linear_regression(x,y)
    visualize_data_and_fit(x,y)