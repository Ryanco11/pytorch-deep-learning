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


# tensot data type
print(ones.dtype)
print(random_tensor.dtype)



# create a range of tensors and tensors-like
one_to_ten = torch.arange(0, 10)
print(one_to_ten)
print(torch.arange(start = 0, end = 1000, step = 50))

# create tensors-like
ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros)



# tensor datatypes
# 1\ tensors not right datatype
# 2\ tensors not right shape
# 3\ tensors not right right device
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                                dtype = None,  # float32 or float16  [https://pytorch.org/docs/stable/tensors.html]
                                device = None, # "cuda"
                                requires_grad = False) # whether or not to track gradients

print(float_32_tensor.dtype)
float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor.dtype)



# int tensor
int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.long)
print(int_32_tensor * float_32_tensor)



# get information from tensor
# tensor.dtype
# tensor.shape
# tensor.device
some_tensor = torch.rand(3, 4)
print(some_tensor.dtype)
print(some_tensor.shape)
print(some_tensor.device)