import torch
from torch import nn

class GraphAttentionLayer(nn.Module):
    def __init__(self, inp, out, slope):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Linear(inp, out, bias=False)
        self.a = nn.Linear(out*2, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(slope)
        self.softmax = nn.Softmax(dim=1)
  
    def forward(self, h, adj):
        Wh = self.W(h)
        Whcat = self.Wh_concat(Wh, adj)
        e = self.leakyrelu(self.a(Whcat).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = self.softmax(attention)
        h_hat = torch.mm(attention, Wh)

        return h_hat
 
    def Wh_concat(self, Wh, adj):
        N = Wh.size(0)
        Whi = Wh.repeat_interleave(N, dim=0)
        Whj = Wh.repeat(N, 1)
        WhiWhj = torch.cat([Whi, Whj], dim=1)
        WhiWhj = WhiWhj.view(N, N, Wh.size(1)*2)

        return WhiWhj
 
class MultiHeadGAT(nn.Module):
    def __init__(self, inp, out, heads, slope):
        super(MultiHeadGAT, self).__init__()
        self.attentions = nn.ModuleList([GraphAttentionLayer(inp, out, slope) for _ in range(heads)])
        self.relu = nn.ReLU()
  
    def forward(self, h, adj):
        heads_out = [att(h, adj) for att in self.attentions]
        out = torch.stack(heads_out, dim=0).mean(0)
    
        return self.relu(out)
 
class GAT(nn.Module):
    def __init__(self, inp, out, heads, slope=0.01):
        super(GAT, self).__init__()
        self.gat1 = MultiHeadGAT(inp, out, heads, slope)
        self.gat2 = MultiHeadGAT(out, out, heads, slope)
  
    def forward(self, h, adj):
        out = self.gat1(h, adj)
        out = self.gat2(out, adj)

        return out