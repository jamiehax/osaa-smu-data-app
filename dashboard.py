import streamlit as st
from preprocess import preprocess
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pdfkit
import os
import tempfile


# example dataframes
df_paths = {
    "Education Data": "data/education_test.csv",
    "Income Data": "data/income_test.csv",
    "Poverty Data": "data/poverty_test.csv"
}
dataframes = {name: preprocess(path) for name, path in df_paths.items()}


# create session states
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'report' not in st.session_state:
    st.session_state.report = None
if 'filters' not in st.session_state:
    st.session_state.filters = []


# UI and logic for choosing a dataset
st.sidebar.header("Select Dataset")
selected_df_name = st.sidebar.selectbox('Select Dataset', list(dataframes.keys()), label_visibility="collapsed")
selected_df = dataframes[selected_df_name]


# title and help
st.title("OSAA SMU's Data Dashboard")
st.markdown("This Data Dashboard allows for quick access to summary statistics about a dataset. Use the sidebar to the left to select which dataset you want to investigate. Once you have selected a dataset, use the *Dataset Filters* section to filter and subset the Dataset to only focus on the area(s) of interest. In that section is where you will also find a brief summary of the dataset. To generate and or download a more detailed summary, go to the *Dataset Profile Report* section once you have selected and filtered the dataset.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

# filter the dataset
st.subheader("Dataset Filters")
st.write("Use the filters in this section to select a subset of the Dataset you are interested in looking at. For example, you can choose to look at only certain countries or only certain columns of the data. Below the filters is a table displaying the summary statistics of the selected dataset.")

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
selected_countries = st.multiselect('select countries:', list(country_codes.keys()))
selected_country_codes = [country_codes[country] for country in selected_countries]

selected_columns = st.multiselect('select columns:', selected_df.columns.tolist(), selected_df.columns.tolist())


# check to see if filters have been updated and update profile state if filters change
filters = [selected_countries, selected_columns]
if filters != st.session_state.filters or selected_df_name != st.session_state.dataset:
    st.session_state.filters = filters
    st.session_state.dataset = selected_df_name
    st.session_state.report = None
    

# summary section
st.markdown("#### Summary")
if not selected_df[selected_columns].empty:
    summary = selected_df[selected_columns].describe()

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


st.markdown("<hr>", unsafe_allow_html=True)
st.write("")


# create the dataframe profile and display it
st.subheader("Dataset Profile Report")
st.write("Click the button below to generate a more detailed report of the filtered dataset. Depending on the size of the selected dataset, this could take some time. Once a report has been generated, it may be downloaded as a PDF document. To do so, follow the steps below the report.")

if st.button('Generate Dataset Profile Report'):
    if selected_df[selected_columns].empty:
        st.write("no data available for the subsetted data.")
        st.session_state.report = None
    else:
        profile = ProfileReport(selected_df[selected_columns], title=f"Profile Report for {selected_df_name}", explorative=True)
        st.session_state.report = profile

if st.session_state.report:
    st_profile_report(st.session_state.report)

st.write("")
st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

# download the generated report as a PDF document
st.subheader("Download the Generated Report as a PDF")
st.markdown("To download the report, it must first be converted to a PDF document. Click the *Convert to PDF* button below to convert the report to a PDF document. Once it has been converted, another button will appear to download it.")
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

        # Clean up temporary files
        os.remove(profile_file_path)
        os.remove(pdf_file_path)

    else:
        st.write("no report has been generated to convert")

