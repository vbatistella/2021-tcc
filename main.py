import os
import numpy as np
import math
from gan import GAN
import pandas as pd

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def generate_dataset(filename):
    # Check if preprocessed Dataset is available
    if os.path.isfile('dataset/movie.npy'):
        return np.load('dataset/movie.npy', allow_pickle=True)

    # Get the ratings file
    ratings = pd.read_csv("dataset/ml-1m/ratings.dat", sep="::", names=["userid", "movieid", "rating", "timestamp"], engine="python")

    # Math for timestamp division
    t_min = ratings["timestamp"].min()
    delta_t = ratings["timestamp"].max() - ratings["timestamp"].min()
    months = int(math.floor(delta_t/(60*60*24*30)))

    # More arguments for matrix building
    users  = ratings["userid"].max()
    movies = ratings["movieid"].max()

    # Log results
    print("timestamps: " + str(months))
    print("movies: "     + str(movies))
    print("users: "      + str(users) )

    # Total time for prediction (y)
    y_time = 24

    # Matrix Creation
    matrix = np.zeros((users, months-y_time, movies))

    for i in range(users):
        user_row = ratings[ratings.userid == i+1]
        for _, row in user_row.iterrows():
            t = math.floor((row["timestamp"]-t_min)/delta_t * months)-1

            # Group on a column all 24 months of ratings
            # This is done to avoid data sparsity
            if t > months-y_time-1:
                t = months-y_time-1

            r = row["rating"]
            m = row["movieid"]-1
            matrix[i][t][m] = r/5

    # Save model for future use
    np.save("dataset/movie", matrix)
    return matrix

if __name__ == '__main__':
    # Build GAN model
    gan = GAN()

    # Generate full dataset for training
    data = generate_dataset("ratings.dat")
    train = data[5000:]
    valid = data[:5000]

    # Log sanity check
    print(data.shape)

    gan.train(100, train, batch_size=64, sample_interval=100)
    gan.validate(valid)