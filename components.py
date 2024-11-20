# standard imports
import streamlit as st
import pandas as pd

# llm data analysis imports
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

# mitosheet imports
from mitosheet.streamlit.v1 import spreadsheet

# pygwalker imports
from pygwalker.api.streamlit import init_streamlit_comm, get_streamlit_html
import streamlit.components.v1 as components

# y-data profile imports
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pdfkit
import os
import tempfile



# create the dataframe profile and display it
@st.fragment
def show_report(df):
    """
        Show the Y-Data Profile report for the passed dataframe.

        REMOVED BECAUSE IT WAS SLOW AND WAS NOT BEING USED.
    """

    button_container = st.container()
    report_container = st.container()
    download_container = st.container()

    try:
        with button_container:
            col1, col2 = st.columns(2)
            with col1:
                if st.button('Generate Dataset Profile Report', use_container_width=True, type="primary", disabled=not (df is not None and not df.empty)):
                    
                    # make profile report
                    profile = ProfileReport(df, title="Profile Report for WorldBank Data", explorative=True)

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



@st.fragment
def df_summary(df):
    """
        Create a variable summary for the dataframe.

        TODO: Add ability to filter the dataframe to a specific indicator and year and country. Right now it will provide summary statistics for a column across all rows, which may include multiple indicators, which can be misleading.
    """

    summary = df.describe()
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




@st.fragment
def llm_data_analysis(df, chat_session_id):
    """
    Display a natural language data anylsis chatbot for the passed dataframe. The message history is specific to the passed chat_session_id.
    """

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        """
        Return the chat history for the passed session id.
        """

        if session_id not in st.session_state.chat_history:
            st.session_state.chat_history[session_id] = InMemoryChatMessageHistory()
        return st.session_state.chat_history[session_id]

    def clear_chat_history(session_id: str) -> None:
        """
        Clear the chat history for the passed session id.
        """
        
        if session_id in st.session_state.chat_history:
            st.session_state.chat_history[session_id].clear()
        if session_id in st.session_state.formatted_chat_history:
            st.session_state.formatted_chat_history[session_id].clear()

    def display_chat_history(session_id: str) -> None:
        """
        Display the chat history in a formatted way.
        """

        if st.session_state.formatted_chat_history.get(session_id, None) is None:
            st.session_state.formatted_chat_history[session_id] = []

        messages = st.session_state.formatted_chat_history.get(session_id, None)
        if messages is not None:
            for message in messages:
                st.chat_message(message["role"]).markdown(message["content"])

    def str_token_counter(text: str) -> int:
        """
        Return the number of tokens to encode the passed text.
        """
        
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))

    def tiktoken_counter(messages: List[BaseMessage]) -> int:
        """
        Return the number of tokens for the message history using the tiktoken counter.
        """
        
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

    def summarize_dataframe(df: pd.DataFrame, max_rows=5, max_categories=25) -> str:
        """
        Return a string combining a summary and preview of the dataframe. 
        """
        
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



    model = AzureChatOpenAI(
        azure_deployment="osaagpt32k",
        api_key=os.gentenv('azure'),
        azure_endpoint="https://openai-osaa-v2.openai.azure.com/",
        openai_api_version="2024-05-01-preview"
    )

    trimmer = trim_messages(
        max_tokens=1000,
        strategy="last",
        token_counter=tiktoken_counter,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    prompt = ChatPromptTemplate.from_messages(
        [
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

    # first trim the message history, then format with the prompt, then send to the model
    chain = (
        RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer) 
        | prompt 
        | model
    )
    config = {"configurable": {"session_id": chat_session_id}}

    # display the formatted message history
    messages_container = st.container()
    with messages_container:
        display_chat_history(chat_session_id)


    with st.container():
        if prompt := st.chat_input("ask about the data..."):

            st.session_state.formatted_chat_history[chat_session_id].append({"role": "user", "content": prompt})

            messages_container.chat_message("user").markdown(prompt)

            if df is not None:
                # df_string = summarize_dataframe(filtered_df)
                df_string = df.to_string()
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

            with messages_container:
                display_chat_history(chat_session_id)




@st.fragment
def show_mitosheet(df):
    """
    Show a Mitosheet for the passed dataframe.
    """

    new_dfs, code = spreadsheet(df)

    if code:
        st.markdown("##### Generated Code:")
        st.code(code)



@st.fragment
def show_pygwalker(df):
    """
    Show the PyGWalker tool for the passed dataframe.
    """
    
    init_streamlit_comm()
    @st.cache_resource
    def get_pyg_html(df: pd.DataFrame) -> str:
        html = get_streamlit_html(df, spec="./gw0.json", use_kernel_calc=True, debug=False)
        return html

    components.html(get_pyg_html(df), width=1300, height=1000, scrolling=True)

