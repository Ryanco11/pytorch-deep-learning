    
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

import torch
from torch import nn
from sklearn.model_selection import train_test_split

import numpy as np


# setup device agonistic code to run cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# calculate accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


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
            self.layer_3 = nn.Linear(in_features=10, out_features=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
            # return self.layer_3(self.layer_2(self.layer_1(x))) # x-> layer_1 -> layer_2 -> output
    

    model_3 = CircleModelV2().to(device)

    print(model_3)
    print(model_3.state_dict())

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(
        model_3.parameters(),
        lr=0.1
    )


    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    epochs = 10000

    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)
    print(next(model_3.parameters()).device)

    for epoch in range(epochs):
        model_3.train()

        y_logits = model_3(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train,
                          y_pred=y_pred)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        ## Testing
        model_3.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model_3(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
            # 2. Calcuate loss and accuracy
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test,
                                    y_pred=test_pred)


        # Print out what's happening
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

    # Make predictions
    model_3.eval()
    with torch.inference_mode():
        y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()
    y_preds[:10], y[:10] # want preds in same format as truth labels

    # Plot decision boundaries for training and test sets
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model_3, X_train, y_train) # model_1 = no non-linearity
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model_3, X_test, y_test) # model_3 = has non-linearity
    plt.show()

    