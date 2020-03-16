from unittest import TestCase
import numpy as np
from utilities.models.vae_torch import VAE


class TestVAE(TestCase):
    def test_generate(self):

        # X = np.random.poisson(lam=10, size=1000)
        # vae = VAE(n_features=1)
        # vae.fit(X)

        data = np.random.randn(1000)
        X = np.array([
            data,
            data*10,
            data*5 + 5
        ]).T
        vae = VAE(n_features=X.shape[1])
        vae.fit(X)


    def test_fit(self):
        self.fail()
