import streamlit as st
from helper_functions import refresh_db


col1, col2 = st.columns(2)
with col1:
    st.image("content/OSAA-Data-logo.svg")


st.title("SMU's Data App")
st.markdown("Welcome to the Office of the Speical Advisor to Africa's Strategic Management Unit's Data App. Use the sidebar to the left or the page links below to navigate between the data products.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

st.markdown("#### Data Dashboard")
st.markdown("Use the Data Dashboard to upload a dataset and get quick access to summary statistics about it, as well as a detailed report with *YData Profiling*. Use the AI chatbot to understand the dataset with natural language questions. Use the Mitosheet Spreadsheet as an embedded spreadsheet to manipulate data and create visualizations. Use the PyGWalker Data Visualization tool to create visualizations from the data.")
st.page_link("dashboard.py", label="Dashboard", icon=":material/analytics:", use_container_width=True)

st.markdown("#### WorldBank Data Dashboard")
st.markdown("Use the WorldBank Data Dashboard for exploratory data analysis of the World Bank's Data. Search available indicators by keyword, and select and download data by indicator, country, and time range. Create automatic interactive time series graphs on the selected data. Use the AI chatbot to understand the dataset with natural language questions. Use the Mitosheet Spreadsheet as an embedded spreadsheet to manipulate data and create visualizations. Use the PyGWalker Data Visualization tool to create visualizations from the data.")
st.page_link("wb_dashboard.py", label="WB Dashboard", icon=":material/bar_chart:", use_container_width=True)

st.markdown("#### SDG Data Dashboard")
st.markdown("Use the SDG Dashboard for exploratory data analysis of the United Nations Sustainable Development Goals DataBase. Explore the 17 sustainable development goals and their corresponding indicators, and select and download data by indicator, country, and time range. Create automatic interactive time series graphs on the selected data. Use the AI chatbot to understand the dataset with natural language questions. Use the Mitosheet Spreadsheet as an embedded spreadsheet to manipulate data and create visualizations. Use the PyGWalker Data Visualization tool to create visualizations from the data.")
st.page_link("sdg_dashboard.py", label="SDG Dashboard", icon=":material/show_chart:", use_container_width=True)

st.markdown("#### Contradictory Analysis Tool")
st.markdown("Use the Contradictory Analysis Tool to check if analysis contradicts any previous in OSAA's publications. This tool uses large language models with retrieval augmented generation and therefore may provide wrong answers, so alwayds double check.")
st.page_link("check_analysis.py", label="Contradictory Analysis Tool", icon=":material/check:", use_container_width=True)


# st.markdown("#### Data Sheet")
# st.markdown("The Data Sheet allows for the automation of excel sheet processes and analysis of an uploaded dataset with *Mitosheet*.")
# st.page_link("datasheet.py", label="Data Sheet", icon=":material/table_chart:", use_container_width=True)

# st.markdown("#### Data Visualization Tool")
# st.markdown("The Data Visualization tool uses *PyGWalker* to automatically create data visualizations like graphs and charts on uploaded datasets.")
# st.page_link("visualizations.py", label="Data Visulization Tool", icon=":material/insert_chart:", use_container_width=True)

# st.markdown("<hr>", unsafe_allow_html=True)
# st.write("")

# st.header("Settings")
# if st.button("refresh database", use_container_width=True, type="primary"):
#     refresh_db(st.session_state.db_path)