import keras
from keras import layers
import numpy as np
from data_helpers import get_sin_sequences
import pandas as pd
import matplotlib.pyplot as plt


def generator_model(n_time_steps, n_time_series):
    """ An example model of a generator

    TODO Try to change n_time_steps to noise dimension

    :param n_time_steps:
    :param n_time_series:
    :return:
    """
    generator_input = keras.Input(shape=(n_time_steps, n_time_series))
    x = layers.LSTM(100, return_sequences=True)(generator_input)
    x = layers.core.Dense(n_time_series)(x)
    generator = keras.models.Model(generator_input, x)
    return generator


def disciminator_model(n_time_steps, n_time_series):
    """ An example discriminator

    :param n_time_steps:
    :param n_time_series:
    :return:
    """
    discriminator_input = layers.Input(shape=(n_time_steps, n_time_series))
    x = layers.LSTM(100)(discriminator_input)
    x = layers.Dense(1, activation='sigmoid')(x)
    discriminator = keras.models.Model(discriminator_input, x)
    return discriminator


class GAN(object):
    """
    A class defining a GAN using keras models
    """
    def __init__(self, n_time_steps, n_time_series, 
                generator_model_func=generator_model, discriminator_model_func=disciminator_model,
                discriminator_optimizer=keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8),
                gan_optimizer=keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
    ):
        """ Initializes the GAN with its models

        :param n_time_steps: Length of sequences to generate
        :param n_time_series: Number of time series to generate in a sample
        :param generator_model_func: function that gets a keras model of the generator
        :param discriminator_model_func: function that gets a keras model of the disciminator
        :param discriminator_optimizer: Keras Optimizer for the discriminator
        :param gan_optimizer: Keras Optimizer for the generator
        """
        self.n_time_series = n_time_series
        self.n_time_steps = n_time_steps

        self.discriminator_optimizer = discriminator_optimizer
        self.gan_optimizer = gan_optimizer

        self.generator_model_func = generator_model_func
        self.discriminator_model_func = discriminator_model_func

        self._build_models()

    def _build_models(self):
        """ Creates the keras models and compiles them
        """
        self.generator = self.generator_model_func(self.n_time_steps, self.n_time_series)

        discriminator = self.discriminator_model_func(self.n_time_steps, self.n_time_series)
        self.discriminator = discriminator
        # To stabilize training, we use learning rate decay
        # and gradient clipping (by value) in the optimizer.
        self.discriminator.compile(optimizer=self.discriminator_optimizer, loss='binary_crossentropy')

        discriminator.trainable = False
        gan_input = keras.Input(shape=(self.n_time_steps, self.n_time_series))
        gan_output = self.discriminator(self.generator(gan_input))
        gan = keras.models.Model(gan_input, gan_output)
        gan.compile(optimizer=self.gan_optimizer, loss='binary_crossentropy')
        self.gan = gan

    def fit(self, data, iterations=30, batch_size=20, plot=True, save_figures=False):
        """ Fits the generator and the discriminator to the sequences of time series

        :param data: np.array (n_sequences, sequence_length, n_time_series)
        :param iterations: Number of iterations to train the model
        :param batch_size: Sample batch size to train on each iteration
        :param plot: If to plot sample sequences during training
        :param save_figures: If to save figures of sample sequences during training
        :return:
        """
        discriminator_losses = []
        generator_losses = []

        indexes = list(range(data.shape[0]))
        # Start training loop
        for step in range(iterations):
            # Sample random points in the latent space
            random_latent_vectors = self.generate_noise(batch_size)

            # Decode them to fake sequences
            generated_seq = self.generator.predict(random_latent_vectors)

            # Combine them with real images
            #start = np.random.randint(0, len(data) - batch_size)
            #stop = start + batch_size
            batch_indexes = np.random.choice(indexes, size=batch_size, replace=True)
            real_seq = data[batch_indexes]
            combined_seq = np.concatenate([generated_seq, real_seq])

            # Assemble labels discriminating real from fake sequences
            labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

            # Add random noise to the labels - important trick!
            # Dont make it to easy for the discriminator.
            labels += 0.05 * np.random.random(labels.shape)

            #print(f'Combined_seq shape:{combined_seq.shape}')
            # Train the discriminator
            d_loss = self.discriminator.train_on_batch(combined_seq, labels)

            # sample random points in the latent space
            random_latent_vectors = self.generate_noise(batch_size)

            # Assemble labels that say "all real sequences"
            misleading_targets = np.zeros((batch_size, 1))

            # Train the generator (via the gan model,
            # where the discriminator weights are frozen)
            # Here we train the generator to fool the disciminator
            g_loss = self.gan.train_on_batch(random_latent_vectors, misleading_targets)

            # Occasionally print, plot or save plots
            if step % 5 == 0:
                # Print metrics
                print('discriminator\tloss at step %s: %s' % (step, d_loss))
                print('generator\tloss at step %s: %s' % (step, g_loss))

            if step % 50 == 0 and plot:                
                generated_seq = self.sample(1)
                print(f'step:{step}, one generated sequences plotted below')
                seq = generated_seq[0]
                plt.figure()
                plt.plot(seq)
                plt.show()

                if save_figures:
                    generated_seqs = self.sample(3)
                    for i in range(3):
                        fig = plt.figure()
                        plt.plot(generated_seqs[i])
                        plt.title(f'Generated at iteration:{step}')
                        plt.savefig(f'plots/generated_sequence_iteration_{step}_plot_{i}.png')
                        plt.close()
                
            generator_losses.append(g_loss)
            discriminator_losses.append(d_loss)

        train_history = pd.DataFrame({
            'generator_loss':generator_losses,
            'discriminator_loss':discriminator_losses})

        return train_history

    def sample(self, n_samples):
        """ Generates samples of sequences

        :param n_samples:
        :return:
        """
        # Sample random points in the latent space
        random_latent_vectors = self.generate_noise(n_samples)

        # Decode them to fake seq
        generated_seq = self.generator.predict(random_latent_vectors)

        return generated_seq

    def generate_noise(self, n_points):
        """ Generates noise to the generator
        :param n_points:
        :return:
        """
        # TODO Try to change n_time_steps to noise dimension
        return np.random.normal(size=(n_points, self.n_time_steps, self.n_time_series))


if __name__=="__main__":
    n_time_steps = 199
    n_points_per_period = 40

    sequences = get_sin_sequences(
        n_points_per_period=n_points_per_period,
        periods=100,
        n_shifts=n_time_steps)

    new_shape = sequences.shape + (1,)
    sequences = sequences.values.reshape(new_shape)
    print(f'Sequences shape:{sequences.shape}')

    gan = GAN(n_time_steps=n_time_steps+1, n_time_series=1)
    hist = gan.fit(sequences, iterations=20, plot=False, save=False)
    print(hist.head())
