import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from model.lstm import LSTM


from torchsummary import summary


# Hyper Parameter
n_epoch = 50
log_interval = 100
batch_num = 128
learning_rate = 0.001

#image size = (1,32,32)

def train(model, epoch, train_loader, optimizer, device='cuda'):
    model.train()
    running_loss = 0.0

    criterion = nn.CrossEntropyLoss().to(device)

    # X:Img, Y:Label
    for batch_id, (X,Y) in enumerate(train_loader):
        X,Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output,Y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_id % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(X), len(train_loader.dataset),
                100. * batch_id / len(train_loader), running_loss/log_interval))
            running_loss =0.0



def test(model,test_loader, device='cuda'):
    model.eval()
    test_loss = 0.0
    correct = 0
    criterion =  nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            loss = criterion(output, Y)
            test_loss +=  loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(Y.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def main():
    gpu_flag = torch.cuda.is_available()
    print("GPU Check :",gpu_flag)
    device = 'cuda' if gpu_flag else 'cpu'
    train_flag = True

    # Augmentation : Random Crop, RandomHorizontal Flip
    # PreProcessing : Noramlization
    train_transform = transforms.Compose([transforms.RandomCrop(28, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.FashionMNIST('./data/train/', transform=train_transform, train=True, download=True)
    testset = torchvision.datasets.FashionMNIST('./data/test/', transform=test_transform, train=True, download=True)

    train_loader = torch.utils.data.DataLoader(trainset,batch_size=batch_num, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

    # torchvision model load 
    # model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    model = LSTM().to(device=device)

    # summary(model,input_size=(3,32,32))


    # Optimizer Select 
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if train_flag:
        for epoch in range(1,n_epoch+1):
            train(model, epoch=epoch, optimizer=optimizer, train_loader=train_loader)
            torch.save(model.state_dict(), "./training_weight/model_"+str(epoch)+".pt")
            test(model,test_loader=test_loader)
    else:
        test(model,test_loader=test_loader)

if __name__ == '__main__':
    main()
