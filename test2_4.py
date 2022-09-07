import pdb
import numpy as np
import sys
sys.path.append('..')
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as lifelines_concordance_index
import matplotlib.pyplot as plt
import math

import torch
from torch import nn
import torch.nn.functional as F
import argparse
from torch.utils.data import TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import train_test_split
from pysurvival.models.simulations import SimulationModel
from pysurvival.models.semi_parametric import CoxPHModel
from pysurvival.utils.metrics import concordance_index
from pysurvival.utils.display import integrated_brier_score

# test to see if it can learn a model simulated high-dimensional covariate and censoring

np.random.seed(seed=0)

#### 2 - Generating the dataset from a Log-Logistic parametric model
# Initializing the simulation model
sim = SimulationModel( survival_distribution = 'log-logistic',
                       risk_type = 'linear',
                       censored_parameter = 10.1,
                       alpha = 0.1, beta=3.2 )

# Generating N random samples 
N = 5000
x_dim = 4
dataset = sim.generate_data(num_samples = N, num_features = x_dim)

# Showing a few data-points 
dataset.head(2)

#### 3 - Creating the modeling dataset
# Defining the features
features = sim.features

# Building training and testing sets #
index_train, index_test = train_test_split( range(N), test_size = 0.2)
data_train = dataset.loc[index_train].reset_index( drop = True )
data_test  = dataset.loc[index_test].reset_index( drop = True )

# Creating the X, T and E input
X_train, X_test = data_train[features].values, data_test[features].values
T_train, T_test = data_train['time'].values, data_test['time'].values
E_train, E_test = data_train['event'].values, data_test['event'].values

#### 4 - Creating an instance of the Cox PH model and fitting the data.
# Building the model
coxph = CoxPHModel()
coxph.fit(X_train, T_train, E_train, lr=0.5, l2_reg=1e-2, init_method='zeros')


#### 5 - Cross Validation / Model Performances
c_index = concordance_index(coxph, X_test, T_test, E_test) #0.92
ch = coxph.predict_cumulative_hazard(X_test[0, :])
risk = coxph.predict_risk(X_test[0, :])
# pdb.set_trace()
print('C-index: {:.4f}'.format(c_index))

ibs = integrated_brier_score(coxph, X_test, T_test, E_test, t_max=10,
            figure_size=(20, 6.5) )
print('IBS: {:.2f}'.format(ibs))



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
            nn.Linear(x_dim + 1, 1000),
            # nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            # nn.BatchNorm1d(1000),
            nn.ReLU(),
            # nn.Linear(1000, 1000),
            # # nn.BatchNorm1d(100),
            # nn.ReLU(),
            nn.Linear(1000, 1),
        )

        # self.net = torch.nn.Sequential(
        #     nn.Linear(1, 1, bias=True)
        # )

        # self.ld = torch.nn.Parameter(torch.Tensor([[0.1]]))

    def forward(self, x, t):
        # t = (t - t_m) / t_std
        # out = self.ld.expand_as(t)
        out = self.net(torch.concat((x, t), dim=-1))

        return out

        # return torch.ones_like(t) * math.log(rate)


def train_one_epoch(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (X, T, E) in enumerate(train_loader):
        X, T, E = X.to(device), T.to(device), E.to(device)
        # data, target = data.to(device), target.to(device)
        # X: [bs, x_dim], T: [bs, 1], E: [bs, 1]
        bs = len(X)
        optimizer.zero_grad()
        
        observed_ll = model(X, T) # [bs, 1]

        # sample some times
        n_time = 1
        sample_time = torch.rand(bs, n_time).cuda() * T # [bs, n_time]
        sample_time = sample_time.reshape(-1, 1) # [bs x n_time, 1]

        X = X.unsqueeze(1) # [bs, 1, x_dim]
        X = X.expand(-1, n_time, -1) # [bs, n_time, x_dim]
        X = X.reshape(-1, X.shape[-1]) # [bs x n_time, x_dim]

        cum_hz = model(X, sample_time) # [bs x n_time, 1]
        cum_hz = cum_hz.reshape(bs, -1) # [bs, n_time]
        cum_hz = torch.exp(cum_hz).mean(-1, keepdim=True) # [bs, 1]

        loss =  -1 * torch.mean(E * observed_ll - cum_hz * T)
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

    return survival.item(), hazard.detach().cpu().numpy()

def predict_cum_hazard(model, x, t, n_time=1000):
    model.eval()
    with torch.no_grad():
        x = x.reshape(1, -1).cuda() # [1, dim]
        x = x.expand(n_time, -1) # [n_time, dim]
        sample_time = torch.linspace(0, t, n_time).cuda().unsqueeze(-1) # [n_time, 1]

        hazard = model(x, sample_time) # [n_time, 1]
        hazard = torch.exp(hazard)
        hazard = hazard.squeeze() # [n_time]
        survival = torch.exp(-1 * hazard.mean() * t)

    return survival.item(), hazard.detach().cpu().numpy()

def get_median(model, x, upper_t, n_time = 1000, n_iter=100):
    # assuming that upper_t is always higher than the median.
    # so S(t) = exp( - cumulative_hazard(t))
    # S(t) is decreasing, cumulative_hazard is increasing.
    # find the point where S(t) = 1/2, or cumulative_hazard(t) = -ln(1/2) = ln(2)

    lower_t = 0.
    for i in range(n_iter):
        sol = (lower_t + upper_t) / 2
        surv, _ = predict_survival_at_t(model, x, sol / T_max)
        if abs(surv - 0.5) < 1e-3:
            break
        if surv > 0.5: # solution is to the left of target
            lower_t = sol
        else:
            upper_t = sol
    return sol


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

# normalize by the max of available time.
T_max = T_train.max()
T_train = T_train / T_max

x_mean = X_train.mean(axis=0, keepdims=True)
x_std = X_train.std(axis=0, keepdims=True)
X_train = (X_train - x_mean) / x_std
print(f'mean={x_mean}, std={x_std}')

dataset1 = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(T_train).unsqueeze(1).float(), torch.from_numpy(E_train).unsqueeze(1).float())
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)

model = Simple(x_dim=X_train.shape[-1]).to(device)
print(model)
optimizer = Adam(model.parameters(), lr=args.lr)

# adding scheduler really helps
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
for epoch in range(1, args.epochs + 1):
    train_one_epoch(args, model, device, train_loader, optimizer, epoch)
    scheduler.step()


# Computing the median time to event of all test datapoints
test_medians = []
for i in range(len(X_test)):
    median = get_median(model, torch.from_numpy((X_test[i, :] - x_mean) / x_std).float(), T_max * 2)
    test_medians.append(median)
# Computing the concordance index
c_index = lifelines_concordance_index(event_times=T_test, predicted_scores=test_medians, event_observed=E_test)
print(f'Concordance index of NN: {c_index:.4f}')


n_plot = 6
fig, ax = plt.subplots(1, n_plot, figsize=(32, 8))

for i in range(n_plot):
    # Randomly extracting a data-point that experienced an event 
    choices = np.argwhere((E_test==1.)&(T_test>=1)).flatten()
    k = np.random.choice( choices, 1)[0]

    # Saving the time of event
    x_test = X_test[k, :]
    t = T_test[k]


    # Computing the Survival function for all times t
    predicted = coxph.predict_survival(x_test).flatten()
    actual = sim.predict_survival(x_test).flatten()

    # Displaying the functions
    ax[i].plot(coxph.times, predicted, color='blue', label='predicted', lw=2)
    ax[i].plot(sim.times, actual, color = 'red', label='actual', lw=2)

    # Actual time
    ax[i].axvline(x=t, color='black', ls ='--')
    ax[i].annotate('T={:.1f}'.format(t), xy=(t, 0.5), xytext=(t, 0.5), fontsize=12)

    # Show everything
    ax[i].set_yticks(np.arange(0, 1, 0.125))
    ax[i].set_ylim([-0.05,1.05])
    ax[i].plot([0, T_max], [0, 0], linestyle='--', alpha=0.5)

    list_survival_rate = []
    for t in np.linspace(0, T_max, 100):
        survival_rate, hazard_rate = predict_survival_at_t(model, torch.from_numpy((x_test - x_mean) / x_std).float(), t / T_max)
        # plt.plot(t, survival_rate, 'rx', markersize=5)
        list_survival_rate.append(survival_rate)
    ax[i].plot(np.linspace(0, T_max, 100), list_survival_rate, markersize=5, linestyle='--', label='nn', lw=2)

    ax[i].legend(fontsize=12)

title = "Comparing Survival functions between Actual and Predicted"
fig.suptitle(title, fontsize=15)
fig.savefig('test2_4.png')