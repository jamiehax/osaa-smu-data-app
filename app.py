import streamlit as st
from helper_functions import setup_db

# create test database
db_path = setup_db()

if 'db_path' not in st.session_state:
    st.session_state.db_path = db_path

home_page = st.Page("home.py", title="Home", icon=":material/home:")
dashboard_page = st.Page("dashboard.py", title="Data Dashboard", icon=":material/analytics:")
wb_dashboard_page = st.Page("wb_dashboard.py", title="WorldBank Data Dashboard", icon=":material/bar_chart:")
sdg_dashboard_page = st.Page("sdg_dashboard.py", title="SDG Data Dashboard", icon=":material/show_chart:")
mitosheet_page = st.Page("datasheet.py", title="Data Sheet", icon=":material/table_chart:")
visualizations_page = st.Page("visualizations.py", title="Data Visualization Tool", icon=":material/insert_chart:")

pg = st.navigation([home_page, dashboard_page, wb_dashboard_page, sdg_dashboard_page])
st.set_page_config(page_title="SMU Data App", page_icon=":material/home:", layout="wide")
pg.run()