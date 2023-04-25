
def main():
    coefficients=[1,0.02,-0.002, 0.014]
    x_values=np.linspace(-10,10,100)
    x,y=generate_synthetic_data(x_values,coefficients)
    degrees = [1, 3, 10]
    models=[fit_polynomial_regression(x,y,degree) for degree in degrees]
   visualize_data_and_fit(x,y,models,degrees)

   