import pdb
import numpy as np
import sys
sys.path.append('..')
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import argparse
from torch.utils.data import TensorDataset
from torch.optim import Adam

breaks=np.arange(0.,365.*5,365./8)
n_intervals=len(breaks)-1
timegap = breaks[1:] - breaks[:-1]

halflife1 = 400
halflife2 = 400
halflife_cens = 400
n_samples=5000
np.random.seed(seed=0)
t1 = np.random.exponential(scale=1 / (np.log(2)/halflife1), size=int(n_samples/2))
t2 = np.random.exponential(scale=1 / (np.log(2)/halflife2), size=int(n_samples/2))
t=np.concatenate((t1, t2))
censtime = np.random.exponential(scale=1 / (np.log(2)/(halflife_cens)), size=n_samples)
f = t<censtime
t[~f] = censtime[~f]
x_train = np.zeros(n_samples)
x_train[int(n_samples/2):]=1.

kmf = KaplanMeierFitter()
kmf.fit(t[0:int(n_samples/2)], event_observed=f[0:int(n_samples/2)])
# plt.plot(breaks,np.concatenate(([1],np.cumprod(y_pred[0,:]))),'bo-')
plt.plot(kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate,color='k')
kmf.fit(t[int(n_samples/2)+1:], event_observed=f[int(n_samples/2)+1:])
# plt.plot(breaks,np.concatenate(([1],np.cumprod(y_pred[-1,:]))),'ro-')
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
rate1 = np.log(2) / halflife1
curve1 = np.exp(- sample_t * rate1)

rate2 = np.log(2) / halflife2
curve2 = np.exp(- sample_t * rate2)

plt.plot(sample_t, curve1)
plt.plot(sample_t, curve2)

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

        self.film = torch.nn.Sequential(
            nn.Linear(x_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )
        self.lin1 = nn.Linear(1, 100)
        self.bn = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(100, 1)

    def forward(self, x, t):
        # for module in self.linears:
        #     x = module(x)
        # x = 1 + F.elu(x, alpha=1.)

        # h = self.film(x)
        out = self.lin1(t)
        out = self.bn(out)
        # out = h[:, [0]] * out + h[:, [1]]
        out = self.relu(out)
        out = self.lin2(out)
        # pdb.set_trace()
        out = 1.001 + F.elu(out, alpha=1.)
        # out = torch.exp(out)
        # out = F.softplus(out)
        return out


def train_one_epoch(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (X, T, E) in enumerate(train_loader):
        X, T, E = X.to(device), T.to(device), E.to(device)
        # data, target = data.to(device), target.to(device)
        # X: [bs, x_dim], T: [bs, 1], E: [bs, 1]
        bs = len(X)
        optimizer.zero_grad()
        
        # input_observed_ll = torch.concat((X, T), dim=-1)
        # observed_ll = torch.log(model(input_observed_ll)).squeeze()
        observed_ll = torch.log(model(X, T)).squeeze()

        # sample some times
        n_time = 100
        sample_time = torch.rand((bs, n_time, 1)).to(device) * T.unsqueeze(1) # [bs, n_time, 1]
        X = X.unsqueeze(1) # [bs, 1, dim]
        X = X.expand(-1, n_time, -1) # [bs, n_time, dim]
        # input_cum_hz = torch.concat((X, sample_time), dim=-1) # [bs, n_time, dim+1]
        # input_cum_hz = input_cum_hz.reshape(-1, X.shape[-1]+1) # [bs x n_time, dim+1]
        # cum_hz = model(input_cum_hz) # [bs, n_time, 1]
        # cum_hz = cum_hz.reshape(bs, n_time) # [bs, n_time]

        cum_hz = model(X.reshape(-1, 1), sample_time.reshape(-1, 1))
        cum_hz = cum_hz.reshape(bs, n_time)

        loss =  -1 * torch.mean(E * observed_ll - cum_hz.mean(dim=-1) * T)
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
        # sample_time = torch.rand((n_time, 1)).cuda() * t # [n_time, 1]
        sample_time = torch.linspace(0, t, n_time).cuda().unsqueeze(-1) # [n_time, 1]
        # input = torch.concat((x, sample_time), dim=-1) # [n_time, dim+1]

        hazard = model(x, sample_time) # [n_time, 1]
        hazard = hazard.squeeze() # [n_time]
        survival = torch.exp(-1 * hazard.mean() * t)

    return survival.item()

# def predict_survival_curve(model, x, t, n_time=1000):
#     model.eval()
#     x = x.reshape(1, -1).cuda() # [1, dim]
#     x = x.expand(n_time, -1) # [n_time, dim]
#     sample_time = torch.linspace(0, t, n_time).cuda().unsqueeze(-1) # [n_time, 1]
#     input = torch.concat((x, sample_time), dim=-1) # [n_time, dim+1]
#     hazard = torch.exp(model(input)) # [n_time, 1]

#     sample_time.squeeze_()
#     hazard.squeeze_()

#     cum_hazard = hazard.cumsum(dim=0) / (torch.arange(n_time).cuda() + 1) * sample_time
#     survival = torch.exp(-1 * cum_hazard * sample_time)

#     return sample_time.detach().cpu().numpy(), survival.detach().cpu().numpy()



# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
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

x_train = (x_train - 0.5)
dataset1 = TensorDataset(torch.from_numpy(x_train).unsqueeze(1).float(), torch.from_numpy(t).unsqueeze(1).float(), torch.from_numpy(f).unsqueeze(1).float())
# dataset2 = TensorDataset(X_test, T_test, E_test)
# train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
# test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.test_batch_size, shuffle=False)

model = Simple(x_dim=1).to(device)
print(model)
# optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
optimizer = Adam(model.parameters(), lr=args.lr)

# scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
for epoch in range(1, args.epochs + 1):
    train_one_epoch(args, model, device, train_loader, optimizer, epoch)
    # scheduler.step()
# test(model, device, test_loader)
# pdb.set_trace()
for t in np.linspace(0, 2000, 10):
    survival_rate = predict_survival_at_t(model, torch.Tensor([[-0.5]]), t)
    plt.plot(t, survival_rate, 'ro', markersize=12)

for t in np.linspace(0, 2000, 10):
    survival_rate = predict_survival_at_t(model, torch.Tensor([[0.5]]), t)
    plt.plot(t, survival_rate, 'rx', markersize=12)

plt.savefig('test2.png')