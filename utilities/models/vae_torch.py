from torch import nn, optim
import torch
from torch.nn import functional as F
from sklearn.model_selection import KFold
import random
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

def real_valued_vae_recon_loss(logvars_decoder, targets, mus_decoder):
    # loss_rec = LOG_2_PI + logvar_x + (x - mu_x)**2 / (2*torch.exp(logvar_x))
    loss_rec = -torch.sum(
        (-0.5 * np.log(2.0 * np.pi))
        + (-0.5 * logvars_decoder)
        + ((-0.5 / torch.exp(logvars_decoder)) * (targets - mus_decoder) ** 2.0),
        dim=(0, 1),
    )
    return loss_rec


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(mu_output, logvar_output, mu_latent, logvar_latent, data):
    # MSE = nn.MSELoss(recon_x, x, reduction='sum')
    #MSE = F.mse_loss(recon_x, x, reduction='sum')
    MSE = real_valued_vae_recon_loss(logvar_output, data, mu_output)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp())
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

        self.hidden_dimension = 10

        self.fc1 = nn.Linear(n_features, 10)
        self.fc21 = nn.Linear(10, 10)
        self.fc22 = nn.Linear(10, self.hidden_dimension)

        self.fc3 = nn.Linear(self.hidden_dimension, 10)
        self.fc41 = nn.Linear(10, n_features)
        self.fc42 = nn.Linear(10, n_features)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc41(h3), self.fc42(h3)

    def forward(self, x):
        mu_latent, logvar_latent = self.encode(x)
        z = self.reparameterize(mu_latent, logvar_latent)
        mu_output, logvar_output = self.decode(z)
        return mu_output, logvar_output, mu_latent, logvar_latent, z

    def generate(self, n):
        z_sample = np.random.randn(n, self.hidden_dimension)
        z_sample = torch.from_numpy(z_sample).float()
        mu_output, mu_logvar = self.decode(z_sample)
        mu_output = mu_output.detach().numpy()
        #mu_logvar = mu_logvar.detach().numpy()
        #mu_std = np.sqrt(np.exp(mu_logvar))
        #mu_output = mu_output*mu_std
        output = self.scaler.inverse_transform(mu_output)
        return output

    def fit(self, X, batch_size=32, epochs=10,
            print_batch=10, lr=1e-4, verbose=False):

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        self.scaler = scaler

        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        n_data = X.shape[0]
        train_loader = DataLoader(X, batch_size=batch_size)

        for epoch in range(1, epochs + 1):
            train_loss = 0
            for batch_idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                mu_output, logvar_output, mu_latent, logvar_latent, z = self(data.float())
                loss = loss_function(mu_output, logvar_output, mu_latent,
                                     logvar_latent, data)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

                if batch_idx % print_batch == 0 and verbose:
                    print('Train Epoch: {} [{}/{} ]\tLoss: {:.6f}'.format(
                        epoch, (batch_idx+1) * batch_size, n_data,
                        loss.item() / len(data)))

            if verbose:
                print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / n_data))