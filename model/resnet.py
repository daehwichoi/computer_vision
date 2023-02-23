import torch.nn as nn
import torch.nn.functional as F

# class Conv_block(nn.Module):

class Resnet(nn.Module):
    def __init__(self, num_classes=10):
        super(Resnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7,7), stride=2, padding=3, bias=False)
        
        self.layer1_r = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=2,bias=False)
        self.layer1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=2, padding=1,bias=False)
        self.layer1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1,bias=False)
        self.layer1_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1,bias=False)
        self.layer1_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1,bias=False)

        self.layer2_r = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=2,bias=False)
        self.layer2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=2, padding=1,bias=False)
        self.layer2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1,bias=False)
        self.layer2_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1,bias=False)
        self.layer2_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1,bias=False)

        self.layer3_r = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), stride=2,bias=False)
        self.layer3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=2, padding=1,bias=False)
        self.layer3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1,bias=False)
        self.layer3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1,bias=False)
        self.layer3_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1,bias=False)

        self.layer4_r = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,1), stride=2)
        self.layer4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=2, padding=1)
        self.layer4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.layer4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.layer4_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)

        self.linear = nn.Linear(512, num_classes)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.bn1_3 = nn.BatchNorm2d(64)
        self.bn1_4 = nn.BatchNorm2d(64)
        self.bn1_5 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(64*2)
        self.bn2_2 = nn.BatchNorm2d(64*2)
        self.bn2_3 = nn.BatchNorm2d(64*2)
        self.bn2_4 = nn.BatchNorm2d(64*2)
        self.bn2_5 = nn.BatchNorm2d(64*2)

        self.bn3_1 = nn.BatchNorm2d(64*2*2)
        self.bn3_2 = nn.BatchNorm2d(64*2*2)
        self.bn3_3 = nn.BatchNorm2d(64*2*2)
        self.bn3_4 = nn.BatchNorm2d(64*2*2)
        self.bn3_5 = nn.BatchNorm2d(64*2*2)

        self.bn4_1 = nn.BatchNorm2d(64*2*2*2)
        self.bn4_2 = nn.BatchNorm2d(64*2*2*2)
        self.bn4_3 = nn.BatchNorm2d(64*2*2*2)
        self.bn4_4 = nn.BatchNorm2d(64*2*2*2)
        self.bn4_5 = nn.BatchNorm2d(64*2*2*2)

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
        x0 = F.relu(self.bn1_1(self.conv1(x)))
        # x = F.max_pool2d(x, stride =2, padding=1, kernel_size = (3,3))
        x = F.relu(self.bn1_2(self.layer1_1(x0)))
        x = F.relu(self.bn1_3(self.layer1_2(x)))
        x1 = F.relu(self.bn1_4(self.layer1_3(x))+self.bn1_5(self.layer1_r(x0)))
        
        x = F.relu(self.bn2_1(self.layer2_1(x1)))
        x = F.relu(self.bn2_2(self.layer2_2(x)))
        x2 = F.relu(self.bn2_3(self.layer2_3(x))+self.bn2_4(self.layer2_r(x1)))

        x = F.relu(self.bn3_1(self.layer3_1(x2)))
        x = F.relu(self.bn3_2(self.layer3_2(x)))
        x3 = F.relu(self.bn3_3(self.layer3_3(x))+self.bn3_4(self.layer3_r(x2)))

        x = F.relu(self.bn4_1(self.layer4_1(x3)))
        x = F.relu(self.bn4_2(self.layer4_2(x)))
        x4 = F.relu(self.bn4_3(self.layer4_3(x))+self.bn4_4(self.layer4_r(x3)))

        # x = F.adaptive_avg_pool2d(x,output_size=(1,1))
        x =x4.view(x4.size(0), -1)
        x = self.linear(x)

        return x


        
