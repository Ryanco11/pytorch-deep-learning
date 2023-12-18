import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)

scalar = torch.tensor(7)
print(scalar)
# tensor(7)
print(scalar.ndim)
# 0
print(scalar.item())
# 7

vector = torch.tensor([7, 7])
print(vector)
# tensor([7, 7])
print(vector.ndim)
# 1
print(vector.shape)
# torch.Size([2])

MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
print(MATRIX)
# tensor([[ 7,  8],
#        [ 9, 10]])
print(MATRIX.ndim)
# 2
print(MATRIX[1])
# tensor([ 9, 10])
print(MATRIX.shape)
# torch.Size([2, 2])

TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9], 
                        [2, 4, 5]]])
print(TENSOR)
# tensor([[[1, 2, 3],
        #  [3, 6, 9],
        #  [2, 4, 5]]])
print(TENSOR.ndim)
# 3
print(TENSOR.shape)
# torch.Size([1, 3, 3])
print(TENSOR[0])
# tensor([[1, 2, 3],
        # [3, 6, 9],
        # [2, 4, 5]])



# create random tensor
random_tensor = torch.rand(3, 4)
print(random_tensor)
print(random_tensor.ndim)

random_tensor = torch.rand(10, 10, 10)
print(random_tensor)


# create zeros and ones tensor
zeros = torch.zeros(size=(3, 4))
print(zeros)

ones = torch.ones(size=(3, 4))
print(ones)


print(ones.dtype)
print(random_tensor.dtype)