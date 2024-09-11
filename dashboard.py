import streamlit as st
from helper_functions import get_dataset_names, get_df
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pdfkit
import os
import tempfile
import pandas as pd
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

# create session states
if 'report' not in st.session_state:
    st.session_state.report = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'formatted_chat_history' not in st.session_state:
    st.session_state.formatted_chat_history = {}

chat_session_id = 'dashboard-id'
CONTEXT_WINDOW = 8192

# title and introduction
st.title("OSAA SMU's Data Dashboard")
st.markdown("The Data Dashboard allows for exploratory data analysis on a dataset through quick access to summary statistics and natural language conversations with an AI chatbot that has the ability to understand the dataset. First select a dataset to view by searching the available datasets by name or uploading your own. Once you have selected a dataset, you can filter and subset the dataset to only focus on the area(s) of interest. Once you have selected and filtered a dataset, you can view the summary statistics on that data. To generate and download a more detailed summary, go to the *Dataset Profile Report* section once you have selected and filtered the dataset. Use the *Natural Language Queries* section to understand the data by asking natural language questions to a chatbot that understands the data.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")


# find and choose a dataset
st.subheader("Select and Filter a Dataset")
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

# filter the dataset
if df is not None:
    with st.container(height=500):
        st.markdown("##### Column Filters")
        
        selected_columns = st.multiselect('select columns to filter:', df.columns.tolist(), df.columns.tolist())
        
        filtered_df = df.copy()
        
        for col in selected_columns:
            st.markdown(f"##### Filter by {col}")
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].isna().all() or df[col].min() == df[col].max():
                    st.markdown(f"cannot filter *{col}* because it has invalid or constant values.")
                else:
                    if pd.api.types.is_integer_dtype(df[col]):
                        min_val, max_val = int(df[col].min()), int(df[col].max())
                        selected_range = st.slider(f"Select range for {col} (int):", min_val, max_val, (min_val, max_val), step=1)
                    elif pd.api.types.is_float_dtype(df[col]):
                        min_val, max_val = float(df[col].min()), float(df[col].max())
                        selected_range = st.slider(f"Select range for {col} (float):", min_val, max_val, (min_val, max_val))
            
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

else:
    filtered_df = None

st.markdown("### Dataset")
if filtered_df is not None and not filtered_df.empty:
    st.write(filtered_df)
else:
    st.write("no dataset selected or the selected filters have resulted in an empty dataset.")


st.markdown("<hr>", unsafe_allow_html=True)
st.write("")
   
# summary section
st.markdown("### Variable Summary")
if filtered_df is not None and not filtered_df.empty:
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
    st.write("no dataset selected or the selected filters have resulted in an empty dataset.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("") 


# natural language dataset exploration
st.subheader("Natural Language Queries")
st.write("Use this chat bot to understand the data with antural language queries. Ask questions in natural language about the data and the chat bot will provide answers in natural language, as well as code (Python, SQL, etc.).")

model = AzureChatOpenAI(
    azure_deployment="gpt35osaa",
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

        if filtered_df is not None:
            df_string = summarize_dataframe(filtered_df)
            df_string = filtered_df.to_string()
        else:
            df_string = "There is no DataFrame available."

        
        num_tokens = tiktoken_counter([HumanMessage(content=df_string)])
        st.write(f"number tokens for used for dataset: {num_tokens}")

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
        st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")


# create the dataframe profile and display it
st.subheader("Dataset Profile Report")
st.write("Click the button below to generate a more detailed report of the filtered dataset. If there is no dataset selcted or the filters have resulted in an empty dataset, the button will be disabled. Depending on the size of the selected dataset, this could take some time. Once a report has been generated, it can be downloaded as a PDF.")

button_container = st.container()
report_container = st.container()
download_container = st.container()

with button_container:
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Generate Dataset Profile Report', use_container_width=True, type="primary", disabled=not (filtered_df is not None and not filtered_df.empty)):
            
            profile = ProfileReport(filtered_df, title=f"Profile Report for {df_name}", explorative=True)

            with report_container:
                with st.expander("show report"):
                    st_profile_report(profile)

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

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

st.subheader("Mitosheet")
if filtered_df is not None and not filtered_df.empty:
    new_dfs, code = spreadsheet(filtered_df)
    if code:
        st.markdown("##### Generated Code:")
        st.write(code)
else:
    st.write("no dataset selected or the selected filters have resulted in an empty dataset.")


st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

st.subheader("Data Visualization Tool")
if filtered_df is not None and not filtered_df.empty:
    init_streamlit_comm()
    @st.cache_resource
    def get_pyg_html(df: pd.DataFrame) -> str:
        html = get_streamlit_html(df, spec="./gw0.json", use_kernel_calc=True, debug=False)
        return html

    components.html(get_pyg_html(df), width=1300, height=1000, scrolling=True)
else:
    st.write("no dataset selected or the selected filters have resulted in an empty dataset.")