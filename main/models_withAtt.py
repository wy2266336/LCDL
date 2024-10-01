import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from GCN_model import *
from attention import *
from utils import *
from torch.nn import CrossEntropyLoss
from transformers import LongformeModel, LongformerTokenizer

    

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(hidden_size,hidden_size)
        self.activition = nn.Tanh()
    
    def forward(self, x):
        first_token = x[:, 0]
        output = self.dense(first_token)
        output = self.activition(output)
        return output

class Encoder(nn.Module):
    def __init__(self, layer, seq_len, N, N_GCN, h, d_model, label_N, label_embedding_dim, threshold, word_embed, adj_file, is_bert=False, is_label_g=True):
        super(Encoder, self).__init__()
        self.is_bert = is_bert
        self.is_label_g = is_label_g
        self.word_embed = word_embed
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.bnorm2d = nn.BatchNorm2d(h)
        self.pooler = Pooler(d_model)
        self.gcn_layers = N_GCN
        self.gcns = nn.ModuleList()
        self.loss = nn.MultiLabelSoftMarginLoss()
        if self.is_label_g:
            self.gc1 = GraphConvolution(label_embedding_dim,d_model)
            self.gc2 = GraphConvolution(d_model,d_model)
            self.relu = nn.LeakyReLU(0.2)

            _adj = generate_A(label_N,threshold,adj_file)
            self.A = nn.Parameter(torch.from_numpy(_adj).float()).to(torch.device('cuda'))
        #else:
        self.mlp = MLP(2*d_model,label_N)

        for i in range(self.gcn_layers):
            gategcn = GatesGraphConvolution(d_model,d_model,h)
            self.gcns.append(gategcn)


    def forward(self, inputs, mask, label_emb):
        batch_size = inputs.size(0)
        break_probs = []
        x = self.word_embed(inputs) #input is input to the word embedding layer, and the output x is [b, s, d_model]
        group_prob = 0.
        weight = 0.

       #EncoderLayer layer, get the output x, and the result of the fusion of attention weight and probability matrix 'attn_weight', the output x is [b, s, d_model], attn_weight is [b, h, s, s]
        for i, layer in enumerate(self.layers):
            x,group_prob,break_prob, attn_weight = layer(x, mask,group_prob)
            break_probs.append(break_prob)
            if i == 3:
                weight = attn_weight

        pooled_output = self.pooler(x)

        weight = self.bnorm2d(weight) #self.cnn(weight)

        break_probs = torch.stack(break_probs, dim=1) 
        
        x = self.norm(x)
        
        adjs = [adj.squeeze(1) for adj in torch.split(weight,1,dim=1)]
        for i in range(self.gcn_layers):
            x = self.gcns[i](x,adjs)

        x = torch.mean(x,dim=1,keepdim=False) 
        
        if self.is_label_g:
           
            pooled_output = pooled_output.unsqueeze(1) 
            label_emb = label_emb[0]
            adj = generate_adj(self.A).detach()
            g_x = self.gc1(label_emb, adj)
            g_x = self.relu(g_x)
            g_x = self.gc2(g_x, adj) 
            g_x = g_x.repeat([batch_size,1,1]) 
            #g_x = g_x.transpose(1,2)
            #x = torch.matmul(x,g_x).squeeze(1)
            p_x = self.label_attention(g_x, pooled_output)
            x = torch.cat([x,p_x],dim=-1) 
       
        x = self.mlp(x)

        return x,break_probs

    def label_attention(self, g_x, x):
        x = x.transpose(1,2) 
        att = torch.matmul(g_x, x) 
        att_score = F.softmax(att, dim=1) 
        att_score = att_score.transpose(1,2)
        att_result = torch.matmul(att_score,g_x) 
        att_result = att_result.squeeze(1) 
        return att_result

    def masked_lm_loss(self, out, y):
        batch_size = out.shape[0]
        label_N = out.shape[1]
        loss = self.loss(out,y)

        out = torch.sigmoid(out)
        
        t = torch.zeros_like(out)
        pre_label = t.masked_fill(out>0.4, 1)
        pre_label = pre_label.view([batch_size*label_N])

       
        y = y.view([batch_size*label_N])
        return loss, pre_label, y


    def next_sentence_loss(self):
        pass


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, group_attn, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.group_attn = group_attn
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, group_prob):
        group_prob,break_prob = self.group_attn(x, mask, group_prob)
        x, attn_weight = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, group_prob, mask))
        return self.sublayer[1](x, self.feed_forward), group_prob, break_prob, attn_weight



class MLP(nn.Module):

    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Linear(input_size//2, input_size//4),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Linear(input_size//4, output_size))

    def forward(self, x):
        out = self.linear(x)
        return out
