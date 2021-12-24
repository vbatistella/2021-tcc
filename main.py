import os
import numpy as np
import math
from gan import GAN
import pandas as pd
from datetime import datetime
import sparse
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

def netflix_dataset(index, step):
    movie = 0

    revised = []

    user_count = 0
    user_dict = {}

    min_date = "1999-11-11"
    max_date = "2005-12-31"
    users = user_count
    users = step
    movies = 17770

    min_date = datetime.strptime(min_date, "%Y-%m-%d")
    max_date = datetime.strptime(max_date, "%Y-%m-%d")
    diff = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month)
    matrix = np.zeros((users, diff-(12*2), movies))

    for i in range(1,5):
        filename = "dataset/archive/combined_data_"+str(i)+".txt"
        # print(filename)
        f = open(filename, 'r')

        while True:
            line = f.readline()
            if not line:
                break

            if ":" in line:
                movie = line[:-2]
                continue

            args = line[:-1].split(",")

            user = int(args[0])
            rating = int(args[1])/5
            date = args[2]
            
            if not user in user_dict:
                user_dict[user] = user_count
                user_count += 1

            if user_dict[user] < step*index and user_dict[user] >= step*(index-1):
                user = user_dict[user] - step*(index-1)

                movie = int(movie)-1
                rating = rating
                date = datetime.strptime(date, "%Y-%m-%d")
                datediff = ((date.year - min_date.year) * 12 + (date.month - min_date.month)) - 1

                if datediff > diff-(12*2)-1:
                    datediff = diff-(12*2)-1

                matrix[user][datediff][movie] = rating

    # print("months:", diff - 12*2)
    print("users :", users)
    # print("movies:", movies)

    return matrix



if __name__ == '__main__':
    # Build GAN model
    gan = GAN()

    train = True
    validate = True

    # Generate full dataset for training
    # data = generate_dataset("ratings.dat")

    user_count = 480000
    step_size = 1000
    training_porcentage = .90
    training_size = math.floor(user_count*training_porcentage)
    steps = training_size // step_size

    if train:
        # gan.load()
        # 39
        for i in range(steps):
            print(i,"/",steps,"(", str(math.floor((i/steps)*10000)/100)+"%",")")
            # data = netflix_dataset(i, step_size)
            data = sparse.load_npz("dataset/netflix_revised/"+str(i)+".npz")
            data = sparse.COO.todense(data)

            # Log sanity check
            # print("shape:", data.shape)

            if i%50 == 0:
                gan.save()

            gan.train(100, data, batch_size=5, sample_interval=50)

    if validate:
        validation = []
        gan.load_generator()
        for i in range(0,20):
            print(i,"/",480)
            # data = netflix_dataset(i, step_size)
            data = sparse.load_npz("dataset/netflix_revised/"+str(i)+".npz")
            data = sparse.COO.todense(data)
            for i in range(5):
                validation.append(gan.validate(data[i*200:(i+1)*200]))
        
        with open("validation.txt", "wb") as fp:
             pickle.dump(validation, fp)

        with open("validation.txt", "rb") as fp:
            a = pickle.load(fp)
        
        a = np.array(a)
        print(np.mean(a))
        print(np.median(a))
