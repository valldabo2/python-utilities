from torch import nn
import torch


def uniform_sampler():
    return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian


class GAN:
    def __init__(self, input_size, output_size, hidden_size=5, noise_size=5):
        self.discriminator = Discriminator(input_size, hidden_size, output_size=1, f=torch.relu)
        self.generator = Generator(noise_size, hidden_size, output_size=input_size, f=torch.relu)
        self.noise_generator = ()

    def sample(self, n_samples):
        noise = self.noise_generator(n_samples)
        return self.generator(noise)


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = torch.sigmoid(x)
        return x
