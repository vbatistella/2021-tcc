import pandas as pd
import matplotlib.pyplot as plt

def get_percentage(df, name, value, column):
    total = df.shape[0]
    count = df[df[column] == value].shape[0]
    return name+": ("+'{0:.2f}'.format(count/total*100)+"%) "+str(count)

def main():
    movies = pd.read_csv("ml-1m/movies.dat", sep="::", names=["movieid", "title", "genres"], engine="python")
    ratings = pd.read_csv("ml-1m/ratings.dat", sep="::", names=["userid", "movieid", "rating", "timestamp"], engine="python")
    users = pd.read_csv("ml-1m/users.dat", sep="::", names=["userid", "gender", "age", "occupation", "zipcode"], engine="python")

    # Sanity check
    # print(movies.head())
    # print(ratings.head())
    # print(users.head())

    # Values count
    print("Totals")
    print("movies: "+str(movies.shape[0]))
    print("ratings: "+str(ratings.shape[0]))
    print("users: "+str(users.shape[0]))
    print()

    # Ratings
    print("Ratings")
    print(get_percentage(ratings, "5 stars", 5, "rating"))
    print(get_percentage(ratings, "4 stars", 4, "rating"))
    print(get_percentage(ratings, "3 stars", 3, "rating"))
    print(get_percentage(ratings, "2 stars", 2, "rating"))
    print(get_percentage(ratings, "1 stars", 1, "rating"))
    print()

    # Users
    print("Users")
    print("Gender")
    print(get_percentage(users, "M", "M", "gender"))
    print(get_percentage(users, "F", "F", "gender"))
    print()

    print("Age")
    print(get_percentage(users, "<18", 1, "age"))
    print(get_percentage(users, "<25", 18, "age"))
    print(get_percentage(users, "<35", 25, "age"))
    print(get_percentage(users, "<45", 35, "age"))
    print(get_percentage(users, "<50", 45, "age"))
    print(get_percentage(users, "<56", 50, "age"))
    print(get_percentage(users, "56+", 56, "age"))
    print()

    print("Profession")
    print(get_percentage(users, "other", 0, "occupation"))
    print(get_percentage(users, "academic/educator", 1, "occupation"))
    print(get_percentage(users, "artist", 2, "occupation"))
    print(get_percentage(users, "clerical/admin", 3, "occupation"))
    print(get_percentage(users, "college/grad student", 4, "occupation"))
    print(get_percentage(users, "customer service", 5, "occupation"))
    print(get_percentage(users, "doctor/health care", 6, "occupation"))
    print(get_percentage(users, "executive/managerial", 7, "occupation"))
    print(get_percentage(users, "farmer", 8, "occupation"))
    print(get_percentage(users, "homemaker", 9, "occupation"))
    print(get_percentage(users, "K-12 student", 10, "occupation"))
    print(get_percentage(users, "lawyer", 11, "occupation"))
    print(get_percentage(users, "programmer", 12, "occupation"))
    print(get_percentage(users, "retired", 13, "occupation"))
    print(get_percentage(users, "sales/marketing", 14, "occupation"))
    print(get_percentage(users, "scientist", 15, "occupation"))
    print(get_percentage(users, "self-employed", 16, "occupation"))
    print(get_percentage(users, "technician/engineer", 17, "occupation"))
    print(get_percentage(users, "tradesman/craftsman", 18, "occupation"))
    print(get_percentage(users, "unemployed", 19, "occupation"))
    print(get_percentage(users, "writer", 20, "occupation"))


if __name__ == "__main__":
    main()