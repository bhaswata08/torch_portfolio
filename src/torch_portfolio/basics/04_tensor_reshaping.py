import torch

x = torch.arange(9)

x_3x3 = x.view(
    3, 3
)  # Use this for NN, faster then reshape, needs tensor to be stored in contiguous memory
x_3x3 = x.reshape(3, 3)  # Safe, but performance loss

y = x_3x3.t()

# print(y.view(9)) # Throws error: RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
print(y.contiguous().view(9))

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=0).shape)
print(torch.cat((x1, x2), dim=1).shape)

z = x1.view(-1)  # Will flatten everything
print(z.shape)


batch = 64
x = torch.rand((batch, 2, 5, 5))
z = x.view(batch, -1)  # will preserve batch dimension
print(z.shape)

#################################
# Old flattening:
z = x1.view(-1)
z = x.view(batch, -1)

# Modern flattening:
z = x1.flatten()  # Flattens everything into a 1D tensor
z = torch.flatten(
    x, start_dim=1
)  # Preserves the batch dimension (dim 0), flattens the rest
#################################


z = x.permute(0, 2, 1, 3)  # Transpose is just a special case of permute function
print(z.shape)

x = torch.arange(10)
print(x.unsqueeze(0).shape)  # 1x10
print(x.unsqueeze(1).shape)  # 10x1

x = torch.arange(10).unsqueeze(0).unsqueeze(1)  # 1x1x10
z = x.squeeze(1)
print(z.shape)
