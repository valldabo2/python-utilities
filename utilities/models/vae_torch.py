from torch import nn, optim
import torch
from torch.nn import functional as F
from sklearn.model_selection import KFold
import random
from torch.utils.data import DataLoader


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # MSE = nn.MSELoss(recon_x, x, reduction='sum')
    MSE = F.mse_loss(recon_x, x, reduction='none')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def batch_sampler(n_data, batch_size):
    all_indexes = set(range(0, n_data))
    while len(all_indexes) > 0:
        if len(all_indexes) > batch_size:
            batch_indexes = random.sample(all_indexes, k=batch_size)
            all_indexes = all_indexes.difference(set(batch_indexes))
            yield list(batch_indexes)
        else:
            yield list(all_indexes)


class VAE(nn.Module):
    def __init__(self, n_features):
        super(VAE, self). __init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc21 = nn.Linear(10, 10)
        self.fc22 = nn.Linear(10, 10)

        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, n_features)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def generate(self, n):
        pass

    def fit(self, X, batch_size=64):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.train()
        train_loss = 0

        n_data = X.shape[0]
        n_batches = int(n_data/batch_size)

        epoch = 1
        train_loader = DataLoader(X, batch_size=batch_size)
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = self(data.float())
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 1 == 0:
                print('Train Epoch: {} [{}/{} ]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * n_batches, n_data,
                           loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / n_data))