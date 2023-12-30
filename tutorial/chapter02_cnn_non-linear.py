    
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

import torch
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    n_samples = 1000
    X, y = make_circles(
        n_samples,
        noise=0.03,
        random_state=42
    )

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    # plt.show()

    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)


    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    print(X_train[:5], y_train[:5])