import torch
from torch import nn 
import matplotlib.pyplot as plt
import numpy as np

print(torch.__version__)


def plot_prediction(train_data=None,
                    train_labels=None,
                    test_data=None,
                    test_labels=None,
                    predictions=None):


    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c='b', s=4, label="Training data")
    plt.scatter(test_data, test_labels, c='g', s=4, label="Testing data")

    if predictions is not None:

        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')

    plt.legend(prop={"size":14})
    plt.show()


if __name__ == "__main__":
    # create device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device :  {device}")

    # create data using the linear regression formula of y = weight * X + bias
    weight = 0.7
    bias = 0.3

    # create range values
    start = 0
    end = 1
    step = 0.02

    # create x and y 
    X = torch.arange(start, end, step)
    y = weight * X + bias
    
    # split data
    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    # plt
    # plot_prediction(train_data=X_train,
                    # train_labels=y_train,
                    # test_data=X_test,
                    # test_labels=y_test,
                    # predictions=None)

    # building pytorch linear model
    class LinearRegressionModelV2(nn.Module):
        def __init__(self):
            super().__init__()
            # use nn.Linear() for creating model parameters
            # self.linear_layer = nn.Linear(in_features=1,
            #                              out_features=1)
            self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
            self.bias = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # return self.linear_layer(x)
            return self.weights * x + self.bias
    # 
    torch.manual_seed(42)
    model_1 = LinearRegressionModelV2()
    print(model_1, model_1.state_dict())

    # check the model current device
    print(next(model_1.parameters()).device)

    # set the model to use the target device
    model_1.to(device)
    print(next(model_1.parameters()).device)

    ### training 
    # 1\ loss function
    loss_fn = nn.L1Loss()

    # 2\ optimizer
    optimizer = torch.optim.SGD(params=model_1.parameters(),
                                lr=0.0001)


    torch.manual_seed(42)
    epochs = 200

    # put data on the target device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # 3\ training loop
    for epoch in range(epochs):
        print(1)
        model_1.train()
        print(2)
        y_pred = model_1(X_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4\ testining loop
        model_1.eval()
        with torch.inference_mode():
            print(3)
            test_pred = model_1(X_test)
            print(4)
            test_loss = loss_fn(test_pred, y_test)
        
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Liss: {loss} | Test loss: {test_loss}")


