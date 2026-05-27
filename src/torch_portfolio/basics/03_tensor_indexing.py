import torch

batch_size = 10
features = 25
x = torch.rand((batch_size, features))
print(x[0].shape)  # x[0, :]
print(x[:, 0].shape)  # First feature over all batches

print(x[2, 0:7].shape)

# Fancy indexing
x = torch.arange(10)
indices = torch.tensor([2, 5, 8])
print(x[indices])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols])

# Advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[x % 2 == 0])

print(torch.where(x > 5, x, x**2))  # if condition meets, return x, else return x**2
print(torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique())
x = torch.rand((5, 5, 5))
print(x.ndim)
print(x.numel())
