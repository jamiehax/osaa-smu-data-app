import pandas as pd
import duckdb
import os
import numpy as np
import streamlit as st
from langchain_community.vectorstores.duckdb import DuckDB
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def preprocess(df):

    # process dataframe as needed here

    return df


# create a test DuckDB and return the path
def setup_db():
    
    DB_PATH = 'content/db.duckdb'
    CSV_DIR = 'content/data'

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

        # create a test dataset
        countries = ["Nigeria", "Kenya", "South Africa", "Ghana", "Egypt", "Morocco", "Ethiopia", "Tanzania", "Uganda", "Algeria"]
        years = list(range(2000, 2024))

        country_data = {
            country: {
                "base_population": np.random.randint(10_000_000, 50_000_000),  # Base population
                "population_growth": np.random.uniform(1.01, 1.03),  # Population growth rate (1-3% annually)
                "base_gdp": np.random.randint(10_000_000_000, 200_000_000_000),  # Base GDP
                "gdp_growth": np.random.uniform(1.02, 1.05),  # GDP growth rate (2-5% annually)
                "total_land": np.random.randint(150_000, 500_000)  # Total land area (constant)
            } for country in countries
        }

        data = []
        for country in countries:
            base_population = country_data[country]["base_population"]
            population_growth = country_data[country]["population_growth"]
            base_gdp = country_data[country]["base_gdp"]
            gdp_growth = country_data[country]["gdp_growth"]
            total_land = country_data[country]["total_land"]
            
            for i, year in enumerate(years):
                population = base_population * (population_growth ** i)
                
                gdp = base_gdp * (gdp_growth ** i) + np.random.uniform(-0.05, 0.05) * base_gdp  # Add some noise
                
                arable_land = np.random.uniform(0.2, 0.4) * total_land  # Arable land is 20-40% of total
                urban_land = np.random.uniform(0.05, 0.15) * total_land  # Urban land is 5-15% of total
                forest_land = total_land - arable_land - urban_land  # Forest land is the remainder
                
                data.append({
                    "Country": country,
                    "Year": year,
                    "Population": int(population),
                    "GDP": int(gdp),
                    "Arable Land (km2)": int(arable_land),
                    "Urban Land (km2)": int(urban_land),
                    "Forest Land (km2)": int(forest_land)
                })

        df_test = pd.DataFrame(data)
        table_name = 'test_dataset'

        # add the test dataset to DuckDB
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_test")
        conn.execute("INSERT INTO metadata (id, name) VALUES (?, ?)", (0, table_name))


        # import all the CSV files into the DuckDB
        if os.path.exists(CSV_DIR):
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
                conn.execute("INSERT INTO metadata (id, name) VALUES (?, ?)", (idx+1, table_name))

        # close the connection
        conn.close()

    return DB_PATH


def refresh_db(db_path):
    """
    Refresh the database (delete existing database and run setup).
    """
    if os.path.isfile(db_path):
        os.remove(db_path)

    setup_db()

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


def make_vectorstore():

    DB_PATH = "content/vectorstore.duckdb"

    # create embedding model
    embedding_model = AzureOpenAIEmbeddings(
        model="smudataembed",
        api_key=st.secrets['azure'],
        azure_endpoint="https://openai-osaa-v2.openai.azure.com/",
    )

    # create DuckDB vectorstore
    conn = duckdb.connect(database=DB_PATH, read_only=False, 
        config={
                "enable_external_access": "false",
                "autoinstall_known_extensions": "false",
                "autoload_known_extensions": "false"
            }
    )

    vectorstore = DuckDB(connection=conn, embedding=embedding_model)

    # add docs to vectorstore
    add_docs(vectorstore)


def add_docs(vectorstore):
    doc_dir_path = "content/rag_documents"
    doc_files = [f for f in os.listdir(doc_dir_path) if f.endswith('.pdf')]
    for doc_file in doc_files:
        doc_path = os.path.join(doc_dir_path, doc_file)
        loader = PyPDFLoader(doc_path)
        doc = loader.load()
        

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(doc)

        for split in splits:
            source_name = os.path.splitext(doc_file)[0]
            page_number = split.metadata.get("page", "Unknown Page")
            if isinstance(page_number, int):
                page_number += 1

            split.metadata = {
                "source": source_name,
                "page": page_number
            }

        vectorstore.add_documents(splits)


def get_retriever(db_path):
    embedding_model = AzureOpenAIEmbeddings(
        model="smudataembed",
        api_key=st.secrets['azure'],
        azure_endpoint="https://openai-osaa-v2.openai.azure.com/",
    )

    conn = duckdb.connect(database=db_path, read_only=False, 
        config={
                "enable_external_access": "false",
                "autoinstall_known_extensions": "false",
                "autoload_known_extensions": "false"
            }
    )

    vectorstore = DuckDB(connection=conn, embedding=embedding_model)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    return retriever