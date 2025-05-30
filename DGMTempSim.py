import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import DGMArchitecture as dgm

def pde_residual(net, x, t, alpha=0.01):
    x.requires_grad_(True)
    t.requires_grad_(True)
    f = net(torch.cat([x, t], dim=1))
    
    df_dt = torch.autograd.grad(f, t, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    df_dx = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    d2f_dx2 = torch.autograd.grad(df_dx, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    
    residual = df_dt - alpha * d2f_dx2
    return residual

def initial_temp_function(x):
    return torch.where((x >= 0.25) & (x <= 0.75), 50, 0)

input_dim = 1
hidden_dim = 50
output_dim = 1
num_layers = 3
mini_batch_size = 100
T = 10

net = dgm.DGMNet(input_dim, hidden_dim, output_dim, num_layers)
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(2000):
    optimizer.zero_grad()
    
    x_domain = torch.rand(mini_batch_size, 1)
    t_domain = torch.rand(mini_batch_size, 1) * T
    x_initial = torch.rand(mini_batch_size, 1)
    t_initial = torch.zeros(mini_batch_size, 1)
    u_initial = initial_temp_function(x_initial)
    x_boundary = torch.bernoulli(torch.full((mini_batch_size, 1), 0.5))
    t_boundary = torch.rand(mini_batch_size, 1) * T
    u_boundary = torch.zeros(mini_batch_size, 1)
    
    loss = torch.mean(
        pde_residual(net, x_domain, t_domain) ** 2
        + (net(torch.cat([x_initial, t_initial], dim=1)) - u_initial) ** 2 
        + (net(torch.cat([x_boundary, t_boundary], dim=1)) - u_boundary) ** 2)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
        
    
x_vals = np.linspace(0, 1, 100)
t_vals = np.linspace(0, 10, 100)

X, T = np.meshgrid(x_vals, t_vals)

X_torch = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1)
T_torch = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1)

with torch.no_grad():
    U_pred = net(torch.cat([X_torch, T_torch], dim=1)).numpy()
    
U_pred = U_pred.reshape(100, 100)

plt.contourf(X, T, U_pred, levels=100, cmap="coolwarm")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")
plt.title("DGM Solution")
plt.show()            