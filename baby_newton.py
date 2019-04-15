import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BabyNewtonNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    num = 10
    g = 10.0
    epoch_size = 10000
    learing_rate = 0.0001 

    # Generate fake free fall time -> distance data
    t_list = []
    d_list = []
    for i in range(num):
        t_list.append(i)
        d_list.append(0.5 * g * i ** 2 )

    t_array = np.asarray(t_list, dtype=np.float32)
    d_array = np.asarray(d_list, dtype=np.float32)

    t_array = t_array.reshape((-1, 1))
    d_array = d_array.reshape((-1, 1))
    assert t_array.shape == (num, 1) and d_array.shape == (num, 1), "The shapes are not as expected!"

    net = BabyNewtonNet()
    t_tensor = torch.from_numpy(t_array)
    d_tensor = torch.from_numpy(d_array)

    criterion = nn.MSELoss()

    optimizer = optim.SGD(net.parameters(), lr=learing_rate)

    for i in range(epoch_size):
        # print(net.fc1.weight.grad)
        output = net(t_tensor)
        loss = criterion(output, d_tensor)
        if (i + 1) % 10 == 0:
            print(i, loss)
            print(output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(d_tensor)