import streamlit as st
from helper_functions import preprocess, get_dataset_names, get_df
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pdfkit
import os
import tempfile



# create session states
if 'df_name' not in st.session_state:
    st.session_state.df_name = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'filters' not in st.session_state:
    st.session_state.filters = []


# title and introduction
st.title("OSAA SMU's Data Dashboard")
st.markdown("This Data Dashboard allows for quick access to summary statistics about a dataset. Use the sidebar to the left to select which dataset you want to investigate. Once you have selected a dataset, use the *Dataset Filters* section to filter and subset the Dataset to only focus on the area(s) of interest. In that section is where you will also find a brief summary of the dataset. To generate and or download a more detailed summary, go to the *Dataset Profile Report* section once you have selected and filtered the dataset.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")


# find and choose a dataset
st.markdown("#### Select a Dataset")
dataset_names = get_dataset_names()
selected_dataset_name = st.selectbox("find a dataset", dataset_names, placeholder="search datasets...", label_visibility="collapsed")
st.session_state.df = get_df(selected_dataset_name)
st.session_state.df_name = selected_dataset_name
st.write(f"Selected Dataset: {st.session_state.df_name}")
 

st.write("")

# filter the dataset
st.markdown("#### Filter Dataset")
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

if st.session_state.df is not None:
    selected_columns = st.multiselect('select columns:', st.session_state.df.columns.tolist(), st.session_state.df.columns.tolist())
else:
    selected_columns = None
   

# summary section
st.markdown("#### Summary")
if st.session_state.df is not None:
    if not st.session_state.df[selected_columns].empty:
        summary = st.session_state.df[selected_columns].describe()

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


# create the dataframe profile and display it
st.subheader("Dataset Profile Report")
st.write("Click the button below to generate a more detailed report of the filtered dataset. Depending on the size of the selected dataset, this could take some time. Once a report has been generated, it may be downloaded as a PDF document. To do so, follow the steps below the report.")

if st.button('Generate Dataset Profile Report'):
    if st.session_state.df[selected_columns].empty:
        st.write("no data available for the subsetted data.")
    else:
        profile = ProfileReport(st.session_state.df[selected_columns], title=f"Profile Report for {selected_dataset_name}", explorative=True)
        st.session_state.report = profile

        with st.expander("show report"):
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