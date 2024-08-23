import streamlit as st
import pandas as pd
import plotly.express as px


# title and introduction
st.title("OSAA SMU's SDG Data Dashboard")

st.markdown("Explore the United Nations Sustainable Development Groups DataBase.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")


BASE_URL = "'https://unstats.un.org/sdgs/UNSDGAPIV5/"