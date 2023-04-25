import matplotlib.pyplot as plt


def visualize_data(x,y):
    plt.scatter(x,y)
    plt.xlabel("Feature (x)")
    plt.ylabel("Target (y)")
    plt.title("Synthetic Data with Polynomial Relationship and Noise")
    plt.show()

    