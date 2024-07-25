import pandas as pd

# example datasets
df_paths = {
    "Education Data": "data/education_test.csv",
    "Income Data": "data/income_test.csv",
    "Poverty Data": "data/poverty_test.csv"
}

def preprocess(df_path):

    # create data fram
    df = pd.read_csv(df_path)

    # process data as needed here

    return df


# placeholder function to simulate dataset query
def search_datasets(query):

    dataframes = {name: preprocess(path) for name, path in df_paths.items()}

    matching_datasets = [ds for ds in dataframes if query.lower() in ds.lower()]
    return matching_datasets


# simulate getting dataset names from database
def get_dataset_names():
    return [name for name in df_paths.keys()]


# simulate getting a dataset by name (or key) from a database
def get_df(name):
    return preprocess(df_paths[name])