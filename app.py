import streamlit as st

home_page = st.Page("home.py", title="Home", icon=":material/home:")
dashboard_page = st.Page("dashboard.py", title="Data Dashboard", icon=":material/analytics:")
mitosheet_page = st.Page("datasheet.py", title="Data Sheet", icon=":material/table_chart:")

pg = st.navigation([home_page, dashboard_page, mitosheet_page])
st.set_page_config(page_title="SMU Data App", page_icon=":material/home:", layout="wide")
pg.run()