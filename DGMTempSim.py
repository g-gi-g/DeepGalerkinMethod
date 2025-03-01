import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class DGMBlock(nn.Module):
    def __init__(self, M, d):
        super(DGMBlock, self).__init__()
        self.Uz = nn.Linear(in_features=d+1, out_features=M, bias=False)
        self.Wz = nn.Linear(in_features=M, out_features=M, bias=False)
        self.bz = nn.Parameter(torch.zeros(M))
        self.Ug = nn.Linear(in_features=d+1, out_features=M, bias=False)
        self.Wg = nn.Linear(in_features=M, out_features=M, bias=False)
        self.bg = nn.Parameter(torch.zeros(M))
        self.Ur = nn.Linear(in_features=d+1, out_features=M, bias=False)
        self.Wr = nn.Linear(in_features=M, out_features=M, bias=False)
        self.br = nn.Parameter(torch.zeros(M))
        self.Uh = nn.Linear(in_features=d+1, out_features=M, bias=False)
        self.Wh = nn.Linear(in_features=M, out_features=M, bias=False)
        self.bh = nn.Parameter(torch.zeros(M))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, S_old):
        Z = self.sigmoid(self.Uz(x) + self.Wz(S_old) + self.bz)
        G = self.sigmoid(self.Ug(x) + self.Wg(S_old) + self.bg)
        R = self.sigmoid(self.Ur(x) + self.Wr(S_old) + self.br)
        H = self.sigmoid(self.Uh(x) + self.Wh(S_old * R) + self.bh)
        return (1 - G) * H + Z * S_old
    
class DGMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(DGMNet, self).__init__()
        self.input_layer = nn.Linear(input_dim+1, hidden_dim)
        self.hidden_layers = nn.ModuleList([DGMBlock(hidden_dim, input_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        S = self.sigmoid(self.input_layer(x))
        for layer in self.hidden_layers:
            S = layer(x, S)
        return self.output_layer(S)



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

net = DGMNet(input_dim, hidden_dim, output_dim, num_layers)
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