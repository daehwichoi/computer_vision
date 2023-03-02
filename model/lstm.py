


import torch.nn as nn
import torch.nn.functional as F

# class Conv_block(nn.Module):

class LSTM(nn.Module):
    def __init__(self, num_classes=10):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(28*28,4*28,2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(4*28,10)

    def _init_weights(self, module):
            if isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=1.0)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)


    def forward(self, x):
        x = x.view(-1,28*28)
        # print(x)
        
        x,(final_state, hidden_state) = self.lstm(x)
        x = self.fc(x)

        return x


        
