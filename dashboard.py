import streamlit as st
from helper_functions import get_dataset_names, get_df
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pdfkit
import os
import tempfile
from pandasai import Agent


# set pandas ai API key
os.environ["PANDASAI_API_KEY"] = ""


# create session states
if 'filters' not in st.session_state:
    st.session_state.filters = []
if 'report' not in st.session_state:
    st.session_state.report = None


# title and introduction
st.title("OSAA SMU's Data Dashboard")
st.markdown("The Data Dashboard allows for quick access to summary statistics about a dataset. First select a dataset to view by searching the available datasets by name. Once you have selected a dataset, you can filter and subset the dataset to only focus on your area(s) of interest. Once you have selected and filtered a dataset, you can view the summary statistics on that data. To generate and download a more detailed summary, go to the *Dataset Profile Report* section once you have selected and filtered the dataset.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")


# find and choose a dataset
st.markdown("#### Select a Dataset")
dataset_names = get_dataset_names(st.session_state.db_path)
df_name = st.selectbox("find a dataset", dataset_names, index=None, placeholder="search datasets...", label_visibility="collapsed")

if df_name is not None:
    df = get_df(st.session_state.db_path, df_name)
else:
    df = None
    st.session_state.report = None

st.success(f"Selected Dataset: {df_name}") 
st.write("")

# filter the dataset
st.markdown("#### Filter Dataset")

country_codes = {
    'Republic of Burundi': 'BDI',	
    'Republic of the Congo': 'COG',
    'Democratic Republic of the Congo': 'COD',	
    'Republic of Djibouti': 'DJI',
    'Federal Democratic Republic of Ethiopia': 'ETH',
    'Republic of Ghana': 'GHA',
    'Republic of Liberia': 'LBR',
    'Republic of Mozambique': 'MOZ',
    'Republic of Namibia': 'NAM',
    'Republic of Niger': 'NER',
    'Federal Republic of Nigeria': 'NGA',
    'Republic of Rwanda': 'RWA',
    'Republic of Senegal': 'SEN',
    'Republic of Sierra Leone': 'SLE',
    'Federal Republic of Somalia': 'SOM',
    'United Republic of Tanzania': 'TZA',
    'Republic of Zambia': 'ZMB',
    'Republic of Zimbabwe': 'ZWE'
}

if df is not None:
    selected_countries = st.multiselect('select countries:', list(country_codes.keys()))
    selected_country_codes = [country_codes[country] for country in selected_countries]
    selected_columns = st.multiselect('select columns:', df.columns.tolist(), df.columns.tolist())
else:
    st.write("no dataset selected")
    selected_columns = None
    selected_countries = None
    selected_country_codes = None
   
# summary section
st.markdown("#### Summary")
if df is not None:
    if not df[selected_columns].empty:
        summary = df[selected_columns].describe()

        columns = summary.columns
        tabs = st.tabs(columns.to_list())

        # return summary statistic if present
        def get_stat(stat):
            return summary.loc[stat, column] if stat in summary.index else "N/A"

        for i, column in enumerate(columns):
            with tabs[i]:
                st.markdown(f"**Count**: {get_stat('count')}")
                st.markdown(f"**Mean**: {get_stat('mean')}")
                st.markdown(f"**Standard Deviation**: {get_stat('std')}")
                st.markdown(f"**Min**: {get_stat('min')}")
                st.markdown(f"**25th Percentile**: {get_stat('25%')}")
                st.markdown(f"**50th Percentile (Median)**: {get_stat('50%')}")
                st.markdown(f"**75th Percentile**: {get_stat('75%')}")
                st.markdown(f"**Max**: {get_stat('max')}")
    else:
        st.write("no data to present summary statistics on.")
else:
    st.write("no dataset selected")


st.markdown("<hr>", unsafe_allow_html=True)
st.write("")


# natural language dataset exploration
st.subheader("Natural Language Queries")
st.write("Use this chat bot to understand the data with antural language queries. Ask questions in natural language about the data and the chat bot will provide answers in natural language, as well as python and SQL code.")

st.markdown("#### Ask about the data:")
query = st.text_input(
    "enter your query",
    label_visibility='collapsed',
    placeholder="enter your query"
)

if df is not None:
    if df[selected_columns].empty:
        st.write("no data available for the subsetted data.")
    else:
        # agent = Agent(df[selected_columns])
        # response = agent.chat(query)
        response = "will add this once we have an API key"
        st.markdown("#### Response:")
        st.write(response)
else:
    st.write("no dataset selected")


st.markdown("<hr>", unsafe_allow_html=True)
st.write("")


# create the dataframe profile and display it
st.subheader("Dataset Profile Report")
st.write("Click the button below to generate a more detailed report of the filtered dataset. Depending on the size of the selected dataset, this could take some time. Once a report has been generated, it can be downloaded as a PDF document in the section below.")

if st.button('Generate Dataset Profile Report'):
    if df is not None:
        if df[selected_columns].empty:
            st.write("no data available for the subsetted data.")
        else:
            profile = ProfileReport(df[selected_columns], title=f"Profile Report for {df_name}", explorative=True)

        with st.expander("show report"):
            st_profile_report(profile)

    else:
        st.write("please select a dataset.")

st.write("")
st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

# download the generated report as a PDF document
st.subheader("Download the Generated Report as a PDF")
st.markdown("To download the report, it must first be converted to a PDF document. Click the *Convert to PDF* button below to convert the report to a PDF document. Once it has been converted, a button will appear below to download it.")
if st.button('Convert to PDF'):

    # check to see if profile has been generated
    if st.session_state.report:

        # retrieve the profile from the session state
        profile = st.session_state.report

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_html:
            profile_file_path = tmp_html.name
            profile.to_file(profile_file_path)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_pdf:
            pdf_file_path = tmp_pdf.name
        
        pdfkit.from_file(profile_file_path, pdf_file_path)

        with open(pdf_file_path, 'rb') as f:
            st.download_button('Download PDF', f, file_name='dataset profile report.pdf', mime='application/pdf')

        # clean up temporary files
        os.remove(profile_file_path)
        os.remove(pdf_file_path)

    else:
        st.write("no report has been generated to convert")