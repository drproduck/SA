import pdb
import numpy as np
import sys
sys.path.append('..')
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import math

import torch
from torch import nn
import torch.nn.functional as F
import argparse
from torch.utils.data import TensorDataset
from torch.optim import Adam

# test to see if it can learn only 1 function

np.random.seed(seed=0)
rate = 0.005
n_samples = 5000
t = np.random.exponential(scale=1. / rate, size=int(n_samples))
t_max = t.max()
x_train = np.ones(n_samples)

kmf = KaplanMeierFitter()
kmf.fit(t, event_observed=np.ones_like(t))
plt.plot(kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate,color='k')
plt.xticks(np.arange(0, 2000.0001, 200))
plt.yticks(np.arange(0, 1.0001, 0.125))
plt.xlim([0,2000])
plt.ylim([0,1])
plt.xlabel('Follow-up time (days)')
plt.ylabel('Proportion surviving')
plt.title('One covariate. Actual=black, predicted=blue/red.')
# plt.show()


# true data generating function
sample_t = np.linspace(0, 2000, 100)
curve = np.exp(- sample_t * rate)

plt.plot(sample_t, curve)

class Simple(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        # self.linears = nn.ModuleList()
        # for i, (in_dim, out_dim) in enumerate(zip(n_dim[:-1], n_dim[1:])):
        #     self.linears.append(nn.Linear(in_dim, out_dim))
        #     if i < len(n_dim)-2:
        #         self.linears.append(nn.ReLU())
        #     if i < len(n_dim)-2:
        #         self.linears.append(nn.BatchNorm1d(out_dim))

        self.net = torch.nn.Sequential(
            nn.Linear(1, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

        # self.net = torch.nn.Sequential(
        #     nn.Linear(1, 1, bias=True)
        # )

        # self.ld = torch.nn.Parameter(torch.Tensor([[0.1]]))

    def forward(self, x, t):
        # t = (t - t_m) / t_std
        # out = self.ld.expand_as(t)
        out = self.net(t)

        return out

        # return torch.ones_like(t) * math.log(rate)


def train_one_epoch(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (X, T) in enumerate(train_loader):
        X, T = X.to(device), T.to(device)
        # data, target = data.to(device), target.to(device)
        # X: [bs, x_dim], T: [bs, 1], E: [bs, 1]
        bs = len(X)
        optimizer.zero_grad()
        
        observed_ll = model(X, T) # [bs, 1]

        # sample some times
        n_time = 100
        sample_time = torch.rand(bs, n_time).cuda() * T # [bs, n_time]
        sample_time = sample_time.reshape(-1, 1) # [bs x n_time, 1]

        X = X.unsqueeze(1) # [bs, 1, x_dim]
        X = X.expand(-1, n_time, -1) # [bs, n_time, x_dim]
        X = X.reshape(-1, X.shape[-1]) # [bs x n_time, x_dim]

        cum_hz = model(X, sample_time) # [bs x n_time, 1]
        cum_hz = cum_hz.reshape(bs, -1) # [bs, n_time]
        cum_hz = torch.exp(cum_hz).mean(-1, keepdim=True) # [bs, 1]

        loss =  -1 * torch.mean(observed_ll - cum_hz * T)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(X), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def predict_survival_at_t(model, x, t, n_time=1000):
    model.eval()
    with torch.no_grad():
        x = x.reshape(1, -1).cuda() # [1, dim]
        x = x.expand(n_time, -1) # [n_time, dim]
        sample_time = torch.linspace(0, t, n_time).cuda().unsqueeze(-1) # [n_time, 1]

        hazard = model(x, sample_time) # [n_time, 1]
        hazard = torch.exp(hazard)
        hazard = hazard.squeeze() # [n_time]
        survival = torch.exp(-1 * hazard.mean() * t)

    return survival.item(), hazard.mean().item()


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

dataset1 = TensorDataset(torch.from_numpy(x_train).unsqueeze(1).float(), torch.from_numpy(t).unsqueeze(1).float())
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)

model = Simple(x_dim=1).to(device)
print(model)
optimizer = Adam(model.parameters(), lr=args.lr)

# scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
for epoch in range(1, args.epochs + 1):
    train_one_epoch(args, model, device, train_loader, optimizer, epoch)

for t in np.linspace(0, 1000, 100):
    survival_rate, hazard_rate = predict_survival_at_t(model, torch.Tensor([[1.]]), t)
    print(hazard_rate)
    plt.plot(t, survival_rate, 'rx', markersize=12)

x = torch.Tensor([[1.]]).cuda()
t = torch.Tensor([[20.]]).cuda()
model.eval()
y = model(x, t)

plt.savefig('test2.png')