import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision


fnames = ["vision_training/imgs/{}.png".format(line.strip().split()[0]) for line in open("vision_training/labels.txt")]
labels = [np.array([float(v) for v in line.strip().split()[1:]]).astype(np.float32) for line in open("vision_training/labels.txt")]
images = [np.array(Image.open(fname))/255 for fname in fnames]
images = [np.transpose(v, (2,0,1)).astype(np.float32) for v in images]
trainloader = torch.utils.data.DataLoader(list(zip(images, labels)), batch_size=8, shuffle=True, num_workers=2)
print("Loaded Data")

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # 160 x 120
    self.pool = nn.MaxPool2d(2, 2) 
    self.conv1 = nn.Conv2d(3, 6, 5)           # 160/120 - 156 --> 78
    self.conv2 = nn.Conv2d(6, 12, 5)          #  78/ 58 -  74 --> 37
    self.conv3 = nn.Conv2d(12, 24, 5)         #  37/ 27 -  33 --> 16
    self.conv4 = nn.Conv2d(24, 48, 5)         #  16/ 11 -  12 --> 6/3
    self.fc1 = nn.Linear(3 * 6 * 48, 3 ** 4)
    self.fc2 = nn.Linear(3 ** 4, 3)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = self.pool(F.relu(self.conv4(x)))
    x = x.view(-1, 3 * 6 * 48)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

net = Net()
crit = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=5e-5)

for epoch in range(10):
  running_loss = 0.0
  for i, data in tqdm(enumerate(trainloader, 0)):
    inputs, labels = data
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = crit(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % 25 == 24:
      print('[{}, {:5d}] loss: {:.3f}'.format(epoch +1, i + 1, running_loss / 25)) 
      running_loss = 0.0
