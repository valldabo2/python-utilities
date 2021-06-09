from unittest import TestCase
import numpy as np
from utilities.models.vae_torch import VAE


class TestVAE(TestCase):
    def test_generate(self):

        n_features = 3
        X = np.random.randn(10000, n_features)
        means = np.array([1, 3  , 5])
        std = np.array([1, 10, 5])
        X = X*std + means
        print(X.mean(axis=0))
        print(X.std(axis=0))

        vae = VAE(n_features=X.shape[1])
        vae.fit(X, batch_size=32, epochs=10, verbose=True)

        sample = vae.generate(1000)
        print(sample.mean(axis=0))
        print(sample.std(axis=0))

    #def test_fit(self):
    #    self.fail()
