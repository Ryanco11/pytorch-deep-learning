    
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
            # return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(X)))))
            return self.layer_2(self.layer_1(x)) # x-> layer_1 -> layer_2 -> output
    

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

    epochs = 1000

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
    plot_decision_boundary(model_1, X_train, y_train) # model_1 = no non-linearity
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model_3, X_test, y_test) # model_3 = has non-linearity
    plot.show()

    