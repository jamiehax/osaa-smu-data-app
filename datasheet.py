import streamlit as st
from helper_functions import get_dataset_names, get_df
from mitosheet.streamlit.v1 import spreadsheet


# example csv files
csv_paths = {
    "Education Data": "data/education_test.csv",
    "Income Data": "data/income_test.csv",
    "Poverty Data": "data/poverty_test.csv"
}


# title and introduction
st.title("OSAA SMU's Data Sheet")

st.markdown("The SMU's Data Sheet allows for the automation of excel sheet processes and analysis with *Mitosheet*.")

# find and choose a dataset
st.markdown("#### Select a Dataset")
dataset_names = get_dataset_names()
selected_dataset_name = st.selectbox("find a dataset", dataset_names, placeholder="search datasets...", label_visibility="collapsed")
st.write(f"Selected Dataset: {selected_dataset_name}")
selected_df = get_df(selected_dataset_name)

new_dfs, code = spreadsheet(selected_df)