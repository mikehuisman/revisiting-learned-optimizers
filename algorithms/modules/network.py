import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Manual implementation of an LSTM cell from
# https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    
    def forward(self, x, hidden):
        """
        x: [n_learner_params, 4]
        """
        # If no hidden and cell state exist yet, initialize them
        if hidden is None:
            # hidden state
            hx = torch.zeros((x.size(0), self.hidden_size), device=torch.cuda.current_device()) # (batch_size, dim_hidden)
            # cell state
            cx = torch.zeros((x.size(0), self.hidden_size), device=torch.cuda.current_device()) # (batch_size, dim_hidden)
        else:
            hx, cx = hidden
        
        gates = self.x2h(x) + self.h2h(hx)

        gates = gates.squeeze()

        # every gate has [n_learner_params, hidden_size]
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        

        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        
        hy = torch.mul(outgate, F.tanh(cy))
        return hy, cy


    def forward_weights(self, x, hidden, weights):
        """
        x: [n_learner_params, 4]
        """
        # If no hidden and cell state exist yet, initialize them
        if hidden is None:
            # hidden state
            hx = torch.zeros((x.size(0), self.hidden_size), device=torch.cuda.current_device()) # (batch_size, dim_hidden)
            # cell state
            cx = torch.zeros((x.size(0), self.hidden_size), device=torch.cuda.current_device()) # (batch_size, dim_hidden)
        else:
            hx, cx = hidden
        
        gates = F.linear(x, weight=weights[0], bias=weights[1]) +\
                F.linear(hx, weight=weights[2], bias=weights[3])

        gates = gates.squeeze()

        # every gate has [n_learner_params, hidden_size]
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        
        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        
        hy = torch.mul(outgate, F.tanh(cy))
        return hy, cy