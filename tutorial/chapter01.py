# pytorch end-to-end workflow

# what_were_covering = {
#     1:"data (prepare and load)",
#     2:"build model",
#     3:"fitting the model to data (training)",
#     4:"making predictions and evaluting a model (inference)",
#     5:"saving and loading a model",
#     6:"putting it all together"
# }


import torch
from torch import nn # nn contains all of pytorch's building blocks for neural networks
import matplotlib.pyplot as plt


print(torch.__version__)



# 1\ Data (prepare and loading)

# Excel
# image
# video
# video
# DNA
# Text

# Machine Learning is a game of two parts
# 1. get data into a numberical representation
# 2. Build a model to learn patters in that numerical representation

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) # unsqueeze : add extra dimensions
y = weight * X + bias

print(X[:10])
print(y[:10])
print(X[:])
print(y[:])

print(len(X))
print(len(y))


# Create a train/test split

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(X_train, y_train, X_test, y_test)



# visualize

def plot_prediction(train_data=X_train,
                    train_labels=y_train,
                    test_data=X_test,
                    test_labels=y_test,
                    predictions=None):



    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c='b', s=4, label="Training data")
    plt.scatter(test_data, test_labels, c='g', s=4, label="Testing data")

    if predictions is not None:
        print(1)

        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')

    plt.legend(prop={"size":14})
    plt.show()

plot_prediction()