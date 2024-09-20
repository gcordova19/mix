# This file is based on the `https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html`.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from moe import MoE

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

net = MoE(input_size=3072, output_size=10, num_experts=10, hidden_size=128, noisy_gating=True, k=4)
#option tranformer
#net = MoE(input_size=3072, output_size=10, num_experts=10, hidden_size=128, noisy_gating=True, k=4, num_heads=2, num_layers=4)

net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

net.train()
for epoch in range(1000):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        inputs = inputs.view(inputs.shape[0], -1)
        outputs, aux_loss = net(inputs)
        loss = criterion(outputs, labels)
        total_loss = loss + aux_loss
        total_loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')


correct = 0
total = 0
net.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs, _ = net(images.view(images.shape[0], -1))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


'''

hp@gaby:~/Documentos/Mixt$ /bin/python3 /home/hp/Documentos/Mixt/nuevo.py
Files already downloaded and verified
Files already downloaded and verified
[1,   100] loss: 2.170
[1,   200] loss: 2.136
[1,   300] loss: 2.123
[1,   400] loss: 2.106
[1,   500] loss: 2.100
[1,   600] loss: 2.089
[1,   700] loss: 2.096
Finished Training
Accuracy of the network on the 10000 test images: 37 %
hp@gaby:~/Documentos/Mixt$ /bin/python3 /home/hp/Documentos/Mixt/nuevo.py
Files already downloaded and verified
Files already downloaded and verified
[1,   100] loss: 2.170
[1,   200] loss: 2.131
[1,   300] loss: 2.120
[1,   400] loss: 2.126
[1,   500] loss: 2.111
[1,   600] loss: 2.089
[1,   700] loss: 2.092
[2,   100] loss: 2.068
[2,   200] loss: 2.077
[2,   300] loss: 2.063
[2,   400] loss: 2.072
[2,   500] loss: 2.069
[2,   600] loss: 2.060
[2,   700] loss: 2.066
[3,   100] loss: 2.052
[3,   200] loss: 2.049
[3,   300] loss: 2.056
[3,   400] loss: 2.045
[3,   500] loss: 2.036
[3,   600] loss: 2.049
[3,   700] loss: 2.037
[4,   100] loss: 2.024
[4,   200] loss: 2.023
[4,   300] loss: 2.038
[4,   400] loss: 2.039
[4,   500] loss: 2.034
[4,   600] loss: 2.044
[4,   700] loss: 2.039
[5,   100] loss: 2.033
[5,   200] loss: 2.030
[5,   300] loss: 2.030
[5,   400] loss: 2.023
[5,   500] loss: 2.014
[5,   600] loss: 2.018
[5,   700] loss: 2.023
[6,   100] loss: 2.011
[6,   200] loss: 2.021
[6,   300] loss: 2.024
[6,   400] loss: 2.025
[6,   500] loss: 2.013
[6,   600] loss: 2.007
[6,   700] loss: 2.018
[7,   100] loss: 2.014
[7,   200] loss: 2.000
[7,   300] loss: 2.008
[7,   400] loss: 2.023
[7,   500] loss: 2.014
[7,   600] loss: 2.004
[7,   700] loss: 2.011
[8,   100] loss: 1.991
[8,   200] loss: 1.998
[8,   300] loss: 2.008
[8,   400] loss: 1.996
[8,   500] loss: 2.005
[8,   600] loss: 2.007
[8,   700] loss: 2.003
[9,   100] loss: 1.996
[9,   200] loss: 1.984
[9,   300] loss: 1.994
[9,   400] loss: 1.995
[9,   500] loss: 1.993
[9,   600] loss: 2.006
[9,   700] loss: 1.996
[10,   100] loss: 1.995
[10,   200] loss: 1.986
[10,   300] loss: 1.995
[10,   400] loss: 2.002
[10,   500] loss: 1.990
[10,   600] loss: 1.993
[10,   700] loss: 1.977
Finished Training
Accuracy of the network on the 10000 test images: 42 %



+epocas 1000

[1000,   100] loss: 1.756
[1000,   200] loss: 1.753
[1000,   300] loss: 1.756
[1000,   400] loss: 1.762
[1000,   500] loss: 1.756
[1000,   600] loss: 1.760
[1000,   700] loss: 1.761
Finished Training
Accuracy of the network on the 10000 test images: 45 %
'''