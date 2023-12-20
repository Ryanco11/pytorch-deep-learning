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

# plot_prediction()




# first pytorch model

# create linear regression model class

# 1\ Gradient descent
# 2\ Baclpropagation

from torch import nn


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                            requires_grad=True,
                                            dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias



# PyTorch model building essentials


# checking the content of our pytorch model

torch.manual_seed(42)    # provide fixed randn seeds for following operations

model_0 = LinearRegressionModel()
print(list(model_0.parameters()))

print(model_0.state_dict())

# Making prediction using 'torch.inference_mode()'



#1
# y_preds = model_0(X_test)

#2
with torch.inference_mode():   ### turns off gradient tracking
    y_preds = model_0(X_test)

#3 
# with torch.no_grad():      # inference_mode is preferred
#     y_preds = model_0(X_test)

print(y_preds)

# plot_prediction(predictions=y_preds)





# Train model

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.001)




epochs = 10000    # An epoch is one loop through the data

print(model_0.state_dict())

for epoch in range(epochs):
    # training mode
    model_0.train()   # train mode set all params that required gradients to required gradients

    # 1 Forward pass
    y_pred = model_0(X_train)

    # 2 Calculate the loss
    loss = loss_fn(y_pred, y_train)
    print(f"Loss :{loss}")

    # 3 optimizer zero grad
    optimizer.zero_grad()

    # 4 perform backpropagation on the loss with respect to the parameters of the model
    loss.backward()

    # 5 step the optimizer (perform gradient descent) 
    optimizer.step()  # by default optimizer changes will acculumate through the loop so... we have to zero them above in step in step 3


    # Testing mode
    model_0.eval() # turns off gradient tracking


print(model_0.state_dict())




with torch.inference_mode():   ### turns off gradient tracking
    y_preds = model_0(X_test)
plot_prediction(predictions=y_preds)
