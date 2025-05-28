import torch
import torch.nn as nn

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