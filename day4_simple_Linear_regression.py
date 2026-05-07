import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x= torch.randn(100,1)
y = 3 * x +1 +0.1 * torch.randn(100,1)


print("shape of x :",x.shape)
print("shape of y:",y.shape)

print("\n first 5 x values :",x[:5].flatten())
print("\n first 5 y values :",y[:5].flatten())


model = nn.Linear(1,1) # 1 input 1 output

print("Model :", model)
print("Starting Weight :", model.weight.item())
print("Starting Bias ", model.bias.item())

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)


losses = []

for epoch in range(100):

    optimizer.zero_grad()

    y_predicted= model(x)

    loss= loss_function(y_predicted,y)

    loss.backward()

    optimizer.step()

    losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch} : loss ={loss.item():.4f}, weight = {model.weight.item():.3f}, bias ={model.bias.item():.3f}")
              



plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.title("Loss going down as the model learn")
plt.savefig("day3_loss_curve.png")
plt.show()