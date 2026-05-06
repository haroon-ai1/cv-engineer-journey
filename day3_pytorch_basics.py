import torch

x = torch.tensor(3.0,requires_grad=True)

y = x**2

print("x :", x)
print("y :",y)

print("\n Calculating Derivative")
y.backward()
print(x.grad)
