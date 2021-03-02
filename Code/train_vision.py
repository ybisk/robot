import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision
import torchvision.models as models
from torchvision import transforms as T

fldr = sys.argv[1]
print("Training with {}".format(fldr))
device = torch. device("cuda:0" if torch. cuda. is_available() else "cpu")

# [-40, 40], [240, 300], [-40, 0]
ranges = np.array([80, 60, 40]).astype(np.float32)
mins = np.array([-40, 240, -40]).astype(np.float32)

def norm_lbl(label):
  label = np.array(label)
  label -= mins     # Shift everything to start at 0
  label /= ranges/2 # Normalize to 2
  label -= 1
  return label.astype(np.float32)

norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
norm_img = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])

fnames = ["{}/imgs/{}.png".format(fldr, line.strip().split()[0]) for line in open("{}/labels.txt".format(fldr))]
labels = [norm_lbl([float(v) for v in line.strip().split()[1:]]) for line in open("{}/labels.txt".format(fldr))]
images = [norm_img(Image.open(fname)) for fname in fnames]
trainloader = torch.utils.data.DataLoader(list(zip(images, labels)), batch_size=8, shuffle=True, num_workers=2)
print("Loaded Data")


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    #self.vpl = models.resnet18(pretrained=True)
    self.vpl = models.squeezenet1_1(pretrained=False)
    self.choice = "squeeze"

    self.fc1 = nn.Linear(512, 128)
    self.fc2 = nn.Linear(128, 32)
    self.fc3 = nn.Linear(32, 3)

  def forward(self, x):
    if self.choice == "squeeze":
      with torch.no_grad():
        # https://pytorch.org/docs/stable/_modules/torchvision/models/squeezenet.html#squeezenet1_1
        x = self.vpl.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
    else: 
      with torch.no_grad():
        # https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet18
        x = self.vpl.maxpool(self.vpl.relu(self.vpl.bn1(self.vpl.conv1(x))))
        x = self.vpl.layer1(x)
        x = self.vpl.layer2(x)
        x = self.vpl.layer3(x)
        x = self.vpl.layer4(x)
        x = self.vpl.avgpool(x)
        x = torch.flatten(x, 1)

    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

net = Net()
net.to(device)
crit = nn.L1Loss() # MSELoss()
optimizer = optim.Adam(net.parameters(), lr=5e-5)

for epoch in range(50):
  running_loss = 0.0
  for i, data in tqdm(enumerate(trainloader, 0)):
    inputs, labels = data
    optimizer.zero_grad()
    outputs = net(inputs.to(device))
    loss = crit(outputs, labels.to(device))
    loss.backward()
    optimizer.step()

    running_loss += loss.cpu().item()
    if i % 25 == 24:
      print('[{}, {:5d}] loss: {:.3f}'.format(epoch +1, i + 1, running_loss / 25)) 
      running_loss = 0.0
      for l, o in zip(labels, outputs.detach().cpu().numpy()):
        print("{:5.2f} {:5.2f} {:5.2f} -- {:5.2f} {:5.2f} {:5.2f}".format(l[0], l[1], l[2], o[0], o[1], o[2]))
