import streamlit as st
from helper_functions import get_dataset_names, get_df
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pdfkit
import os
import tempfile
import pandas as pd
from pandasai import SmartDataframe, Agent
from pandasai.llm import AzureOpenAI



# create session states
if 'report' not in st.session_state:
    st.session_state.report = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# function to display chat history
def display_chat_history():
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**AI:** {message['content']}")


# title and introduction
st.title("OSAA SMU's Data Dashboard")
st.markdown("The Data Dashboard allows for quick access to summary statistics about a dataset. First select a dataset to view by searching the available datasets by name. Once you have selected a dataset, you can filter and subset the dataset to only focus on your area(s) of interest. Once you have selected and filtered a dataset, you can view the summary statistics on that data. To generate and download a more detailed summary, go to the *Dataset Profile Report* section once you have selected and filtered the dataset.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")


# find and choose a dataset
st.subheader("Select a Dataset")
st.write("Either search through existing datasets or upload your own dataset as a CSV or Excel file.")


st.markdown("##### Search Datasets")
dataset_names = get_dataset_names(st.session_state.db_path)
df_name = st.selectbox("find a dataset", dataset_names, index=None, placeholder="search datasets...", label_visibility="collapsed")

if df_name is not None:
    df = get_df(st.session_state.db_path, df_name)
else:
    df = None
    st.session_state.report = None


st.markdown("##### Upload a Dataset (CSV or excel)")
uploaded_df = st.file_uploader("Choose a file", type=["csv", "xlsx"], label_visibility="collapsed")

if uploaded_df is not None:
    df_name = uploaded_df.name
    if uploaded_df.name.endswith('.csv'):
        df = pd.read_csv(uploaded_df)
    elif uploaded_df.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_df)


if df is not None: st.write(df)
st.write("")


st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

# filter the dataset
st.markdown("#### Filter Dataset")
if df is not None:
    with st.expander("show filters:"):
        st.markdown("##### Column Filters")
        
        selected_columns = st.multiselect('select columns to filter:', df.columns.tolist(), df.columns.tolist())
        
        filtered_df = df.copy()
        
        for col in selected_columns:
            st.markdown(f"##### Filter by {col}")
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].isna().all() or df[col].min() == df[col].max():
                    st.markdown(f"cannot filter *{col}* because it has invalid or constant values.")
                else:
                    min_val, max_val = float(df[col].min()), float(df[col].max())
                    selected_range = st.slider(f"select range for {col}:", min_val, max_val, (min_val, max_val))
                    filtered_df = filtered_df[(filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])]
            
            elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                if df[col].isna().all() or df[col].nunique() == 1:
                     st.markdown(f"cannot filter *{col}* because it has invalid or constant values.")
                else:
                    unique_vals = df[col].dropna().unique().tolist()
                    selected_vals = st.multiselect(f"Select values for {col}:", unique_vals, unique_vals)
                    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
            
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                if df[col].isna().all() or df[col].min() == df[col].max():
                     st.markdown(f"cannot filter *{col}* because it has invalid or constant values.")
                else:
                    min_date, max_date = df[col].min(), df[col].max()
                    selected_dates = st.date_input(f"Select date range for {col}:", [min_date, max_date])
                    filtered_df = filtered_df[(df[col] >= pd.to_datetime(selected_dates[0])) & (df[col] <= pd.to_datetime(selected_dates[1]))]
            
            else:
                st.write(f"Unsupported column type for filtering: {df[col].dtype}")
    

        filtered_df = filtered_df[selected_columns]

        if filtered_df.empty:
            st.write("The filters applied have resulted in an empty dataset. Please adjust your filters.")
        else:
            st.markdown("### Filtered Data")
            st.write(filtered_df)
else:
    st.write("No dataset selected")
    filtered_df = None


st.markdown("<hr>", unsafe_allow_html=True)
st.write("")
   
# summary section
st.markdown("### Summary")
if filtered_df is not None:
    if not filtered_df.empty:
        summary = filtered_df.describe()

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

text_col, send_col = st.columns(2)
with text_col:
    query = st.text_input(
        "enter your query",
        label_visibility='collapsed',
        placeholder="enter your query"
    )

with send_col:
    if st.button('Send'):
        if filtered_df is not None:
            if filtered_df.empty:
                st.write("No data available for the subsetted data.")
            else:
                try:
                    azure = AzureOpenAI(
                        api_token=st.secrets['azure'],
                        azure_endpoint="https://openai-osaa-v2.openai.azure.com/",
                        api_version="2024-05-01-preview",
                        deployment_name="gpt35osaa"
                    )

                    smart_df = SmartDataframe(filtered_df, config={"llm": azure})
                    response = smart_df.chat(query)
                    
                    # agent = Agent(smart_df)
                    # response = agent.chat(query)

                    st.session_state.chat_history.append({'role': 'user', 'content': query})
                    st.session_state.chat_history.append({'role': 'ai', 'content': response})
                                        
                except Exception as e:
                    st.error(e)

with st.expander("show conversation"):
    if st.button('clear conversation'):
        st.session_state.chat_history = []
    
    if filtered_df is not None:
        if filtered_df.empty:
            st.write("No data available for the subsetted data.")
        else:
            display_chat_history()
    else:
        st.write("No dataset selected")


st.markdown("<hr>", unsafe_allow_html=True)
st.write("")


# create the dataframe profile and display it
st.subheader("Dataset Profile Report")
st.write("Click the button below to generate a more detailed report of the filtered dataset. Depending on the size of the selected dataset, this could take some time. Once a report has been generated, it can be downloaded as a PDF document in the section below.")

if st.button('Generate Dataset Profile Report'):
    if filtered_df is not None:
        if filtered_df.empty:
            st.write("no data available for the subsetted data.")
        else:
            profile = ProfileReport(filtered_df, title=f"Profile Report for {df_name}", explorative=True)
            st.session_state.report = profile

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