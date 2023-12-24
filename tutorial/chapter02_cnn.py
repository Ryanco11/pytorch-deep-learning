



### 1/ make classification data and get it ready

import sklearn
from sklearn.datasets import make_circles

# 
n_samples = 1000
# 
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

print(len(X), len(y))
print(f"First 5 samples of X:\n {X[:5]}")
print(f"First 5 samples of X:\n {y[:5]}")
# First 5 samples of X:
#  [[ 0.75424625  0.23148074]
#  [-0.75615888  0.15325888]
#  [-0.81539193  0.17328203]
#  [-0.39373073  0.69288277]
#  [ 0.44220765 -0.89672343]]
# First 5 samples of X:
#  [1 1 1 1 0]


# make dataframe of circle data
import pandas as pd
circles = pd.DataFrame({"X1":X[:, 0],
                        "X2":X[:, 1],
                        "label":y})
print(circles.head(10))


# visualize
import matplotlib.pyplot as plt
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
# plt.show()


## check input and output shapes
print(X.shape, y.shape)
#(1000, 2) (1000,)

# view the first example of feature and labels 
X_sample = X[0]
y_sample = y[0]

print(f"values for one sample of X:{X_sample} and the same for y:{y_sample}")
print(f"shapes for one sample of X:{X_sample.shape} and the same for y:{y_sample.shape}")
# values for one sample of X:[0.75424625 0.23148074] and the same for y:1
# shapes for one sample of X:(2,) and the same for y:()


## turn data into tensors and create train and test splits
# turn data into tensors
import torch
print(torch.__version__)

print(type(X), X.dtype)
# <class 'numpy.ndarray'>,  float64

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(X[:5], y[:5])

# tensor([[ 0.7542,  0.2315],
#         [-0.7562,  0.1533],
#         [-0.8154,  0.1733],
#         [-0.3937,  0.6929],
#         [ 0.4422, -0.8967]])   ,    tensor([1., 1., 1., 1., 0.])

print(type(X), type(y), X.dtype, y.dtype)
# <class 'torch.Tensor'>,  <class 'torch.Tensor'>,  torch.float32,  torch.float32


# split data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,   # 0.2 = 20% test, 80% train
                                                    random_state=42)

print(len(X_train), len(X_test), len(y_train), len(y_test))
#800 200 800 200



### 2/ building a model

# setup device agonistic code to run cuda
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


# construct a model
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
    
    def forward(self, x):
        return self.layer_2(self.later_1(x)) # x-> layer_1 -> layer_2 -> output
    
# define a loss function and optimizer

model_0 = CircleModelV0().to(device)
print(model_0)
# CircleModelV0(
#   (layer_1): Linear(in_features=2, out_features=5, bias=True)
#   (layer_2): Linear(in_features=5, out_features=1, bias=True)
# )

print(next(model_0.parameters()).device)
# cuda:0


# create a training and test loop
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

print(model_0.state_dict())

with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f"Length of prediction: {len(untrained_preds)}, Shapes: {untrained_preds.shape}")
print(f"Length of test samples: {len(X_test)}, Shapes: {X_test.shape}")
print(f"First 10 of predictions: {untrained_preds[:10]}")
print(f"First 10 of labels: {y_test[:10]}")



# what loss function you gonna use for each 
    # regression : MAE \ MSE
    # classification : binary cross entropy loss function

# optimizer
    # 


# loss_fn = nn.BCELoss() 
loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

print(model_0.state_dict())


# calculate accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc



### train model

model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
print(y_logits)
# tensor([[0.5863],
#         [0.5001],
#         [0.7513],
#         [0.5398],
#         [0.6546]], device='cuda:0')

print(y_test[:5])
# tensor([1., 0., 1., 0., 1.])

# prediction probailities
y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)
# tensor([[0.4352],
#         [0.4313],
#         [0.4675],
#         [0.4260],
#         [0.5029]], device='cuda:0')

# predicted label
y_preds = torch.round(y_pred_probs)
print(y_preds)
# tensor([[0.],
#         [0.],
#         [0.],
#         [0.],
#         [1.]], device='cuda:0')

# logits -> pred probs -> pred labels
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# check fot equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))
# tensor([True, True, True, True, True], device='cuda:0')

print(y_preds.squeeze())
# tensor([1., 1., 0., 1., 1.], device='cuda:0')