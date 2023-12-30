    
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

if __name__ == "__main__":

    n_samples = 1000
    X, y = make_circles(
        n_samples,
        noise=0.03,
        random_state=42
    )

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.show()