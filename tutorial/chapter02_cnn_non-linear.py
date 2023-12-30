    
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

import torch
from torch import nn
from sklearn.model_selection import train_test_split


# setup device agonistic code to run cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

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


    class CircleModelV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=2, out_features=10)
            self.layer_2 = nn.Linear(in_features=10, out_features=10)
            self.layer_3 = nn.Linear(in_features=10, out_features=10)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(X)))))
    

    model_3 = CircleModelV2().to(device)

    print(model_3)
