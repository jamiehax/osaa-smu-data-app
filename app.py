import streamlit as st
from helper_functions import setup_db

# create test database
db_path = setup_db()

if 'db_path' not in st.session_state:
    st.session_state.db_path = db_path

home_page = st.Page("home.py", title="Home", icon=":material/home:")
dashboard_page = st.Page("dashboard.py", title="Data Dashboard", icon=":material/analytics:")
mitosheet_page = st.Page("datasheet.py", title="Data Sheet", icon=":material/table_chart:")

pg = st.navigation([home_page, dashboard_page, mitosheet_page])
st.set_page_config(page_title="SMU Data App", page_icon=":material/home:", layout="wide")
pg.run()