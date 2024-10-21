import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pdfkit
import os
import tempfile
import plotly.express as px
import requests
from mitosheet.streamlit.v1 import spreadsheet
from pygwalker.api.streamlit import init_streamlit_comm, get_streamlit_html
import streamlit.components.v1 as components
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
import tiktoken
from langchain_core.messages import BaseMessage, ToolMessage
from typing import List


@st.cache_data
def get_data(url):
    """
    Function to get data from the passed URL through an HTTPS request and return it as a JSON object. Data is cached so that function does not rerun when URL doesn't change.
    """
    try:
        data = requests.get(url).json()
    except Exception as e:
        st.cache_data.clear()
        data = e

    return data

# chatbot functions
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = InMemoryChatMessageHistory()
    return st.session_state.chat_history[session_id]

def clear_chat_history(session_id):
    if session_id in st.session_state.chat_history:
        st.session_state.chat_history[session_id].clear()
    st.session_state.formatted_chat_history = {}

def display_chat_history(session_id):
    messages = st.session_state.formatted_chat_history.get(session_id, None)
    if messages is None:
        st.session_state.formatted_chat_history[session_id] = []
        intro_message = "Hi! I am a chatbot assistant trained to help you understand your data. Ask me questions about your currently selected dataset in natural language and I will answer them!"
        st.chat_message("assistant").markdown(intro_message)
        st.session_state.formatted_chat_history[session_id].append({"role": "assistant", "content": intro_message})
    else:   
        for message in messages:
            st.chat_message(message["role"]).markdown(message["content"])

def str_token_counter(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))

def tiktoken_counter(messages: List[BaseMessage]) -> int:
    num_tokens = 3 
    tokens_per_message = 3
    tokens_per_name = 1
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported messages type {msg.__class__}")
        num_tokens += (
            tokens_per_message
            + str_token_counter(role)
            + str_token_counter(msg.content)
        )
        if msg.name:
            num_tokens += tokens_per_name + str_token_counter(msg.name)
    return num_tokens

def summarize_dataframe(df, max_rows=5, max_categories=25):
    summary = df.describe().to_string()
    preview = df.head(max_rows).to_string()

    categorical_counts = []
    non_numeric_columns = df.select_dtypes(exclude='number').columns
    for col in non_numeric_columns:
        counts = df[col].value_counts().nlargest(max_categories).to_string()
        categorical_counts.append(f"Column '{col}' top {max_categories} categories: {counts}")

    categorical_unique = []
    non_numeric_columns = df.select_dtypes(exclude='number').columns
    for col in non_numeric_columns:
        num_unique = df[col].nunique()
        num_missing = df[col].isna().sum()
        categorical_unique.append(f"Column '{col}': {num_unique} unique values and {num_missing} missing values")

    return f"DataFrame Preview (first {max_rows} rows):\n{preview}\n\nDataFrame Numeric Column Summary:\n{summary}\n\nDataFrame non-numeric Column Top Category Counts: {','.join(categorical_counts)}\n\nDataFrame non-numeric Column Unique Values: {','.join(categorical_unique)}"



# read in iso3 code reference df
iso3_reference_df = pd.read_csv('content/iso3_country_reference.csv')
iso3_reference_df['m49'] = iso3_reference_df['m49'].astype(str)

chat_session_id = 'acled-dashboard-chat-id'

# create session states
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'formatted_chat_history' not in st.session_state:
    st.session_state.formatted_chat_history = {}
if 'acled_df' not in st.session_state:
    st.session_state['acled_df'] = None

# title and introduction
st.title("OSAA SMU's ACLED Data Dashboard")

st.markdown("The ACLED Data Dashboard allows for exploratory data analysis of the ACLED Data.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

st.markdown("#### Select Countries")

# select by region
region_mapping = {
    "Western Africa": 1,
    "Middle Africa": 2,
    "Eastern Africa": 3,
    "Southern Africa": 4,
    "Northern Africa": 5,
    "South Asia": 7,
    "Southeast Asia": 9,
    "Middle East": 11,
    "Europe": 12,
    "Caucasus and Central Asia": 13,
    "Central America": 14,
    "South America": 15,
    "Caribbean": 16,
    "East Asia": 17,
    "North America": 18,
    "Oceania": 19,
    "Antarctica": 20
}
selected_regions = st.multiselect("select regions", region_mapping.keys(), None, placeholder="select by region", label_visibility="collapsed")
selected_region_codes = [region_mapping.get(selected_region) for selected_region in selected_regions]

# select by country
country_to_iso_map = dict(zip(iso3_reference_df['Country or Area'], iso3_reference_df['m49']))
selected_countries = st.multiselect("select countries", iso3_reference_df['Country or Area'], None, placeholder="select by country",label_visibility="collapsed")
selected_country_codes = [country_to_iso_map.get(selected_country) for selected_country in selected_countries]

st.markdown("#### Select Time Range")

# select years
selected_years = st.slider( "Select a range of years:", min_value=1960, max_value=2024, value=(1960, 2024), step=1, label_visibility="collapsed")

if st.button("Get Data", type="primary", use_container_width=True):

    # construct API request URL

    api_key = st.secrets['acled_key']
    # email = "james.hackney@un.org"
    email = st.secrets['acled_email']

    BASE_URL = f"https://api.acleddata.com/acled/read?key={api_key}&email={email}"

    region_param = "&region=" + ":OR:region=".join([str(code) for code in selected_region_codes])
    country_param = "&iso=" + ":OR:iso=".join(selected_country_codes)
    year_param = f"&year={selected_years[0]}|{selected_years[1]}&year_where=BETWEEN"

    data_url = f"{BASE_URL}{country_param}{year_param}{region_param}"

    # API query parameters
    data = get_data(data_url)

    st.markdown("#### Dataset")
    if isinstance(data, Exception):
        st.error(data)
    else:
        try:
            df = pd.DataFrame(data['data'])
            if df.empty:
                st.session_state.acled_df = None
                st.markdown("**No data available for the selected countries, regions, and years.**")
            else:
                st.session_state.acled_df = df
                st.write(df)
        except Exception as e:
            st.error(e)
            st.session_state.acled_df = None
        
else:
    df = None
    st.session_state.acled_df = df


st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

st.markdown("### Variable Summary")
@st.fragment
def show_summary():
    """
    Show Summary statistics on variables.
    """

    if st.session_state.acled_df is not None:
        if not st.session_state.acled_df.empty:

            summary = st.session_state.acled_df.describe()

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
        st.write("data not available for the selected indicator(s), countries, and year(s).")
show_summary()


st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

# st.markdown("### Natural Language Analysis")
@st.fragment
def show_chatbot():
    st.write("Use this chat bot to understand the data with natural language questions. Ask questions about the data and the chat bot will provide answers in natural language, as well as code (Python, SQL, etc.).")

    model = AzureChatOpenAI(
        azure_deployment="osaagpt32k",
        api_key=st.secrets['azure'],
        azure_endpoint="https://openai-osaa-v2.openai.azure.com/",
        openai_api_version="2024-05-01-preview"
    )

    trimmer = trim_messages(
        max_tokens=1000, # model max context size is 8192
        strategy="last",
        token_counter=tiktoken_counter,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            # (
            #     "system",
            #     "You are a helpful data analyst assistant. Answer the user's question about their data. You will not have access to the entire dataset, instead you will get the first 5 rows of the data, as well as summaries of the columns. Use this to infer the answers to the users questions.",
            # ),
            (
                "system",
                "You are a helpful data analyst assistant. Answer the user's question about their data.",
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "human",
                "Here is the Pandas DataFrame: {dataframe}."
            ),
            (
                "human",
                "My question is: {prompt}."
            )
        ]
    )

    chain = RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer) | prompt | model

    config = {"configurable": {"session_id": chat_session_id}}


    messages_container = st.container(height=500)
    with messages_container:
        display_chat_history(chat_session_id)


    with st.container():
        if prompt := st.chat_input("ask about the data..."):

            st.session_state.formatted_chat_history[chat_session_id].append({"role": "user", "content": prompt})

            messages_container.chat_message("user").markdown(prompt)

            if st.session_state.acled_df is not None:
                df_string = summarize_dataframe(st.session_state.acled_df)
                df_string = st.session_state.acled_df.to_string()
            else:
                df_string = "There is no DataFrame available."

            
            # num_tokens = tiktoken_counter([HumanMessage(content=df_string)])
            # st.write(f"number tokens for used for dataset: {num_tokens}")

            # get reponse
            with_message_history = RunnableWithMessageHistory(
                chain,
                get_session_history,
                input_messages_key="messages",
            )
            response_generator = with_message_history.stream(
                {
                    "messages": [HumanMessage(content=prompt)],
                    "dataframe": df_string,
                    "prompt": prompt
                },
                config=config
            )
            
            with messages_container:
                with st.chat_message("assistant"):
                    try:
                        response = st.write_stream(response_generator)
                    except Exception as e:
                        response = f"I'm sorry I could not answer your question an error occured. \n\n {e}"
                        st.write(response)

            st.session_state.formatted_chat_history[chat_session_id].append({"role": "assistant", "content": response})


        if st.button("clear chat history", type="primary", use_container_width=True):
            clear_chat_history(chat_session_id)
# show_chatbot()


# st.markdown("<hr>", unsafe_allow_html=True)
# st.write("")

st.markdown("### Dataset Profile Report")
@st.fragment
def show_report():
    st.write("Click the button below to generate a more detailed report of the filtered dataset. If there is no dataset selcted or the filters have resulted in an empty dataset, the button will be disabled. Depending on the size of the selected dataset, this could take some time. Once a report has been generated, it can be downloaded as a PDF.")

    button_container = st.container()
    report_container = st.container()
    download_container = st.container()

    try:
        with button_container:
            col1, col2 = st.columns(2)
            with col1:
                if st.button('Generate Dataset Profile Report', use_container_width=True, type="primary", disabled=not (st.session_state.acled_df is not None and not st.session_state.acled_df.empty)):
                    
                    # make profile report
                    profile = ProfileReport(st.session_state.acled_df, title="Profile Report for UNSDG Data", explorative=True)

                    # display profile report
                    with report_container:
                        with st.expander("show report"):
                            st_profile_report(profile)

                    # download the file
                    with download_container:
                        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_html:
                            profile_file_path = tmp_html.name
                            profile.to_file(profile_file_path)
                        
                        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_pdf:
                            pdf_file_path = tmp_pdf.name
                        
                        pdfkit.from_file(profile_file_path, pdf_file_path)

                        with open(pdf_file_path, 'rb') as f:
                            st.download_button('Download PDF', f, file_name='dataset profile report.pdf', mime='application/pdf', use_container_width=True, type="primary")

                        # clean up temporary files
                        os.remove(profile_file_path)
                        os.remove(pdf_file_path)

            with col2:
                with st.popover("What are YData Profile Reports?", use_container_width=True):
                    st.write("YData Profiling is a Python package that offers a range of features to help with exploratory data analysis. It generates a detailed report that includes descriptive statistics for each variable, such as mean, median, and standard deviation for numerical data, and frequency distribution for categorical data. It will also highlights missing values, detects duplicate rows, and identifies potential outliers. Additionally, it provides correlation matrices to explore relationships between variables, interaction plots to visualize dependencies, and alerts to flag data quality issues like high cardinality or skewness. It also includes visualizations like scatter plots, histograms, and heatmaps, making it easier to spot trends and or anomalies in your dataset.")
    except Exception as e:
        st.error(f"Error generating report:\n\n{e}")
show_report()


st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

st.markdown("### Mitsheet Spreadsheet")
@st.fragment
def show_mitosheet():
    if st.session_state.acled_df is not None and not st.session_state.acled_df.empty:
        new_dfs, code = spreadsheet(st.session_state.acled_df)
        if code:
            st.markdown("##### Generated Code:")
            st.code(code, language='python')
    else:
        st.write("data not available for the selected indicator(s), countries, and year(s).")
show_mitosheet()

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

st.markdown("### PyGWalker Data Visualization Tool")
@st.fragment
def show_pygwalker():
    if st.session_state.acled_df is not None and not st.session_state.acled_df.empty:
        init_streamlit_comm()
        @st.cache_resource
        def get_pyg_html(df: pd.DataFrame) -> str:
            html = get_streamlit_html(st.session_state.acled_df, spec="./gw0.json", use_kernel_calc=True, debug=False)
            return html

        components.html(get_pyg_html(st.session_state.acled_df), width=1300, height=1000, scrolling=True)
    else:
        st.write("data not available for the selected indicator(s), countries, and year(s).")
show_pygwalker()