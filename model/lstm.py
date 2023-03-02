import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_dim, seq, hidden_dim, num_classes=10):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim,batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim,num_classes)

    def _init_weights(self, module):
            if isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=1.0)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)


    def forward(self, x):
        x = x.view(-1,28,28)        
        out,(final_state, hidden_state) = self.lstm(x)
        x = self.fc(final_state[0,:,:])

        return x


        
