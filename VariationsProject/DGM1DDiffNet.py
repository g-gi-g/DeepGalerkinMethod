import torch
import torch.optim as optim
import DGMArchitecture as dgm
import time
import ModifAdagrad

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
hidden_dim = 100
output_dim = 1
num_layers = 20
mini_batch_size = 100
T = 10

net = dgm.DGMNet(input_dim, hidden_dim, output_dim, num_layers)
#optimizer = optim.Adam(net.parameters(), lr=0.001)
#optimizer = optim.Adagrad(net.parameters(), lr=1)
optimizer = ModifAdagrad.ModifAdagrad(net.parameters(), lr=0.1, mr=0.1)

start_time = time.time()

for epoch in range(1000):

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

end_time = time.time()

print(f"Loss = {loss.item():.6f}")
print(f"Time = {end_time - start_time:.6f}")