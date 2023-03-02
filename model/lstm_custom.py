

import torch
import torch.nn as nn
import torch.nn.functional as F


class lstm(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.f_t = nn.Linear(input_size, 4*hidden_size)
        self.h_t = nn.Linear(hidden_size, 4*hidden_size)

        self.apply(self._init_weights)
    def _init_weights(self, module):
            if isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=1.0)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, x, cx):
        hidden_state, cell_state = cx
        x = x.view(-1, x.size(-1))

        x = self.f_t(x) + self.h_t(hidden_state)
        i_g, f_g, c_g, o_g = torch.chunk(x,4,1)
        
        i_g = F.sigmoid(i_g)
        f_g = F.sigmoid(f_g)
        c_g = F.sigmoid(c_g)
        o_g = F.sigmoid(o_g)

        cy = torch.mul(cell_state, f_g) + torch.mul(i_g, c_g)
        hy = torch.mul(o_g, F.tanh(cy))

        return hy,(hy, cy)




class LSTM(nn.Module):
    def __init__(self, input_dim, seq, hidden_dim, num_classes=10):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq = seq

        self.fc = nn.Linear(self.hidden_dim,num_classes)
        self.lstm = lstm(self.input_dim,self.hidden_dim)
        self.num_layer = 28

        self.apply(self._init_weights)
        
    def _init_weights(self, module):
            if isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=1.0)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)


    def forward(self, x):
        h0 = torch.Tensor(torch.zeros(x.size(0), self.hidden_dim)).to('cuda')
        c0 = torch.Tensor(torch.zeros(x.size(0), self.hidden_dim)).to('cuda')

        x = x.view(x.size(0),self.seq,self.input_dim)

        hn = h0
        cn = c0
        
        for seq in range(self.num_layer):
            out,(hn, cn) = self.lstm(x[:,seq], (hn, cn))
        
        x = self.fc(hn)

        return x


        
