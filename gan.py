from tensorflow import keras
import numpy as np
from sklearn.metrics import mean_squared_error

split = 9

class GAN():
    # Build and test generator
    def generator(self):

        model = keras.models.Sequential()
        # TanH instead of ReLu for GPU performance
        model.add(keras.layers.LSTM(256, activation='tanh', recurrent_activation='sigmoid', input_shape=(split,3952,)))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(1*3952, activation='relu'))
        model.add(keras.layers.Reshape((1,3952)))

        noise = keras.layers.Input(shape=(split,3952,))
        seq = model(noise)

        return keras.models.Model(noise, seq)

    # Build and test Discriminator   
    def discriminator(self):

        model = keras.models.Sequential()
         # TanH instead of ReLu for GPU performance
        model.add(keras.layers.LSTM(256, activation='tanh', recurrent_activation='sigmoid', input_shape=(1,3952,)))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        seq = keras.layers.Input(shape=(1,3952))
        validity = model(seq)

        return keras.models.Model(seq, validity)
    
    def __init__(self):
        optimizer = keras.optimizers.Adam(0.0002, 0.5)

        # Build discriminator
        self.discriminator = self.discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.generator()

        # generator generates sequence
        z = keras.layers.Input(shape=(split,3952,))
        gen_seqs = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # discriminator see if valid
        valid = self.discriminator(gen_seqs)

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = keras.models.Model(z , [gen_seqs, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[0.999, 0.001], optimizer=optimizer)

    def train(self, epochs, data, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train = data[:,:split,:]
        y_train = data[:,split:,:]

        data = {}

        # Truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            #
            #    DISCRIMINATOR
            #

            # random sequences
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_seqs = X_train[idx]
            y_seqs = y_train[idx]

            # batch of new sequences
            gen_seqs = self.generator.predict(real_seqs)

            # train discriminator
            d_loss_real = self.discriminator.train_on_batch(y_seqs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #
            #    GENERATOR
            #

            # train generator
            g_loss = self.combined.train_on_batch(real_seqs, [y_seqs, valid])

            # Intervel of epochs for logging
            if epoch % sample_interval == 0:
                print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
            
        print("Saving...")
        self.generator.save("model.h5")
        print("Saved.")

    def validate(self, data):
        model = keras.models.load_model("model.h5")

        x = data[:,:split,:]
        y = data[:,split:,:]

        total = 0
        for user in range(len(y)):
            y_real = y[user][0]

            y_fake = model(x[user].reshape(1, x[user].shape[0], x[user].shape[1]))
            y_fake = np.array(y_fake)[0][0]

            total += mean_squared_error(y_real, y_fake)

        total /= len(y)
        print("mean mse:", total)
