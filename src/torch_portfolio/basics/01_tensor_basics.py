import torch

my_tensor = torch.tensor(
    [[1, 2, 3], [4, 5, 6]], dtype=torch.bfloat16, device="cuda", requires_grad=True
)

print(my_tensor)  # Shape(2, 3), ndim: 2

x = torch.empty(size=(3, 3))
print(x)
x = torch.eye(5, 5)  # Identity matrix
print(x)
x = torch.arange(start=0, end=5, step=1)
print(x)
x = torch.diag(3 * torch.ones(3))  # diagonalization of a matrix
print(x)
