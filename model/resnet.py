import torch.nn as nn


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, **kwargs):
        super(Conv_block, self).__init__()
        self.relu = nn.Relu()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs) 
    

class Res_block(nn.Module):
    def __init__(self) -> None:
        super(Res_block, self).__init__()
        self.relu = nn.Relu()
        self.conv = nn.Conv2d(in_channels=in_)


class Resnet(nn.Module):
    def __init__(self) -> None:
        super()
        pass
    def forword(self):


