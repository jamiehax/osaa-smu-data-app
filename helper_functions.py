import pandas as pd
import duckdb
import os


def preprocess(df):

    # process dataframe as needed here

    return df


# create a test DuckDB and return the path
def setup_db():
    
    DB_PATH = 'db.duckdb'
    CSV_DIR = 'data'

    # check to see if database already exists
    if not os.path.exists(DB_PATH):

        # connect to database, and make it if it doesn't already exist
        conn = duckdb.connect(database=DB_PATH, read_only=False)

        # create a metadata table to store dataset names and primary keys
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """)

        # import all the CSV files into the DuckDB
        csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
        for idx, csv_file in enumerate(csv_files):

            # turn CSV's into DataFrames
            df = pd.read_csv(os.path.join(CSV_DIR, csv_file))

            # import dataframe into DuckDB with the filename as the table name
            table_name = os.path.splitext(csv_file)[0]

            # check to see if it exists already
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            
            # Add an entry to the metadata table
            conn.execute("INSERT INTO metadata (id, name) VALUES (?, ?)", (idx, table_name))


        # Close the connection
        conn.close()

    return DB_PATH


# simulate getting dataset names from database
def get_dataset_names(db_path):

    # connect to database
    conn = duckdb.connect(database=db_path, read_only=True)

    # query - this may change depending on the schema of the database
    query = "SELECT name FROM metadata"
    result = conn.execute(query).fetchall()
    conn.close()

    return [row[0] for row in result]


# simulate getting a dataset by name (or key) from a database
def get_df(db_path, name):

    # connect to database
    conn = duckdb.connect(database=db_path, read_only=True)

    # query and convert to dataframe - this may change depending on the schema of the database
    query = f"SELECT * FROM {name}"
    df = conn.execute(query).df()
    conn.close()

    return preprocess(df)