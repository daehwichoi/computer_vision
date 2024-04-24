import torch
import torch.nn as nn
from torch.utils.data import dataloader

import torchvision
import torchvision.transforms as transforms

from vit_pytorch import ViT

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10("./data/train", train=True, transform=train_transform, download=True)
    testset = torchvision.datasets.CIFAR10("./data/test", train=False, transform=test_transform, download=True)

    train_loader = dataloader.DataLoader(trainset, shuffle=True, batch_size=128)
    test_loader = dataloader.DataLoader(testset, shuffle=False, batch_size=1000)

    model = ViT(image_size=32,
                patch_size=8,
                num_classes=10,
                dim=128,
                depth=3,
                heads=16,
                mlp_dim=1024,
                dropout=0.1,
                emb_dropout=0.1)
    model.to(device)

    optim = torch.optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    log_interval = 10
    n_epoch = 10
    for epoch in range(n_epoch):
        for batch_id, (X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)

            optim.zero_grad()
            output = model(X)


            loss = criterion(output, Y)
            loss.backward()
            optim.step()

            running_loss += loss.item()

            if batch_id % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_id * len(X), len(train_loader.dataset),
                    100. * batch_id / len(train_loader), running_loss/log_interval))
                running_loss =0.0


    model.eval()
    test_loss = 0.0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            loss = criterion(output, Y)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(Y.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))