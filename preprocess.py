import pandas as pd

def preprocess(df_path):

    # create data fram
    df = pd.read_csv(df_path)

    # process data as needed here

    return df


if __name__ == '__main__':
    preprocess('data/education_test.csv')