from datetime import datetime
import numpy as np
import sparse

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
    # print("users :", len(user_dict))
    # print("movies:", movies)

    return matrix

# User total: 480.189, arredondando para 480.000
def main():
    for i in range(0,480):
        print(str((i/480)*100)+'%')
        ds = netflix_dataset(i, 1000)
        ds = sparse.COO.from_numpy(ds)
        sparse.save_npz("dataset/netflix_revised/"+str(i), ds)

main()
