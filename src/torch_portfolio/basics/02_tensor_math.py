import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

z = x / y
# print(z)

# INPLACE operation
t = torch.ones(3)
t.add_(x)
# print(t)

# Matmul
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = x1 @ x2
# print(x3)

# Matrix exp
matrix_exp = 3 * torch.rand(3, 3)
# print(matrix_exp)
# print(matrix_exp**3)  # elem wise A^3
print(torch.linalg.matrix_power(matrix_exp, 3))  # A @ A @ A

# dot product
z = torch.dot(x, y)
# print(z)

# Batch mm
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = tensor1 @ tensor2  # (batch , n, p)
# print(out_bmm)

x_rand = torch.rand((3, 3))
values, indices = torch.max(x_rand, dim=0)
print("x:", x_rand)
print("Values:", values)
print("indices:", indices)
y_rand = torch.rand((3, 3))
print(torch.eq(x_rand, y_rand))
print(torch.sort(y, dim=0, descending=False))


print(torch.clamp(x, min=0, max=10))  # ReLU is just torch.clamp(x, min = 0)

x = torch.tensor([1, 0, 1, 1, 1], dtype=bool)
z = torch.any(x)
z = torch.all(x)
print(z)
