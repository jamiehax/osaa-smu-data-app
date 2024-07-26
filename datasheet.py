import streamlit as st
from helper_functions import get_dataset_names, get_df
from mitosheet.streamlit.v1 import spreadsheet


# title and introduction
st.title("OSAA SMU's Data Sheet")

st.markdown("The SMU's Data Sheet allows for the automation of excel sheet processes and analysis with *Mitosheet*.")

# find and choose a dataset
st.markdown("#### Select a Dataset")
dataset_names = get_dataset_names(st.session_state.db_path)
df_name = st.selectbox("find a dataset", dataset_names, index=None, placeholder="search datasets...", label_visibility="collapsed")

if df_name is not None:
    df = get_df(st.session_state.db_path, df_name)
else:
    df = None

st.success(f"Selected Dataset: {df_name}") 


if df is not None:
    new_dfs, code = spreadsheet(df)