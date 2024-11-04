import streamlit as st
import wbgapi as wb
import pandas as pd
import plotly.express as px
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


# cached functions for retreiving data
@st.cache_data
def get_databases():
    try:
        data = [(database["id"], database["name"]) for database in wb.source.list()]
    except Exception as e:
        data = e
    
    return data

@st.cache_data
def get_indicators():
    try:
        data = list(wb.series.list())
    except Exception as e:
        data = e
    
    return data

@st.cache_data
def get_query_result(search_query, db):
    try:
        data = list(wb.series.list(q=search_query, db=db))
    except Exception as e:
        data = e
    
    return data

@st.cache_data
def get_countries():
    try:
        data = list(wb.economy.list())
    except Exception as e:
        data = e
    
    return data
    
def get_countries_by_region(region):
    return iso3_reference_df[iso3_reference_df['Region Name'] == region]['iso3'].tolist()



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

# create session states
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'formatted_chat_history' not in st.session_state:
    st.session_state.formatted_chat_history = {}
if 'wb_df' not in st.session_state:
    st.session_state['wb_df'] = None
if 'wb_df_melted' not in st.session_state:
    st.session_state['wb_df_melted'] = None

chat_session_id = 'wb-dashboard-chat-id'

# title and introduction
st.title("OSAA SMU's World Bank Data Dashboard")

st.markdown("The WorldBank Data Dashboard allows for exploratory data analysis of the World Bank's Data. First, select one of the WorldBank's databases to use. Then select the indicators, countries, and time range to get data for. Indicators can be filtered with keywords. For example, if you are using the *Doing Business* database and interested in indicators related to construction, enter *construction* into the search box to limit the indicators to only those that mention construction.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

# read in iso3 code reference df
iso3_reference_df = pd.read_csv('content/iso3_country_reference.csv')

st.markdown("#### Select Database:")
databases = get_databases()
if isinstance(databases, Exception): 
    st.write(f"error getting databases info: \n{databases}")
else:
    selected_db_name = st.selectbox("available databases:", [database[1] for database in databases], label_visibility="collapsed")
    selected_db = next(database[0] for database in databases if database[1] == selected_db_name)
    if selected_db: wb.db = selected_db

st.markdown("#### Select Indicators:")
st.write("NOTE: keyword queries ignore the parenthetical part of the indicator name. For example, 'GDP' will not match 'Gross domestic savings (% of GDP)'. To search the parenthetical part too, add an exclamation point like this: '!GDP'")
search_query = st.text_input(
        "enter keywords to filter indicators",
        label_visibility='collapsed',
        placeholder="enter keywords to filter indicators"
    )
query_result = get_query_result(search_query, selected_db)
if isinstance(query_result, Exception): 
    st.write(f"error getting query result info: \n{query_result}")
else:
    formatted_indicators = [f"{indicator['id']} - {indicator['value']}" for indicator in query_result]
    selected_indicator_names = st.multiselect("available indicators:", formatted_indicators, label_visibility="collapsed", placeholder="select indicator(s)")
    selected_indicators = [indicator.split(' - ')[0] for indicator in selected_indicator_names]


st.markdown("#### Select Countries:")

countries = get_countries()
if isinstance(countries, Exception): 
    st.write(f"error getting country info: \n{countries}")
else:
    regions = iso3_reference_df['Region Name'].dropna().unique()

    selected_regions = st.multiselect(
        "select regions:",
        ['SELECT ALL'] + list(regions),
        label_visibility="collapsed",
        placeholder="select by region"
    )

    if 'SELECT ALL' in selected_regions:
        selected_regions = [r for r in regions if r != 'SELECT ALL']
    else:
        selected_regions = [r for r in selected_regions if r != 'SELECT ALL']

    selected_countries = []
    for region in selected_regions:
        selected_countries.extend(get_countries_by_region(region))

    # remove duplicates
    selected_countries = list(set(selected_countries))
    selected_country_names = iso3_reference_df[iso3_reference_df['iso3'].isin(selected_countries)]['Country or Area'].tolist()
    iso3_to_name = dict(zip(iso3_reference_df['iso3'], iso3_reference_df['Country or Area']))
    selected_countries_formatted = [f"{country_code} - {iso3_to_name[country_code]}" for country_code in selected_countries]

    available_countries = list(zip(iso3_reference_df['iso3'].tolist(), iso3_reference_df['Country or Area'].tolist()))
    available_countries_formatted = [f"{country[0]} - {country[1]}" for country in available_countries]

    selected_countries = st.multiselect(
        "Available countries:",
        available_countries_formatted,
        default=selected_countries_formatted,
        label_visibility="collapsed",
        placeholder="select by country"
    )
    selected_iso3_codes = [entry.split(' - ')[0] for entry in selected_countries]

st.markdown("#### Select Time Range")

selected_years = st.slider( "Select a range of years:", min_value=1960, max_value=2024, value=(1960, 2024), step=1, label_visibility="collapsed")

# get world bank data
try:
    if st.button("get data", type='primary', use_container_width=True):
        wb_df = wb.data.DataFrame(selected_indicators, selected_iso3_codes, list(range(selected_years[0], selected_years[1]))).reset_index()

        # deal with edge cases where there is no economy column
        if 'economy' not in wb_df.columns:
            wb_df['economy'] = selected_iso3_codes[0]

        # deal with edge cases where there is no series column
        if 'series' not in wb_df.columns:
            wb_df['series'] = selected_indicators[0]
    
        # add country reference codes
        wb_df = wb_df.merge(iso3_reference_df[['Country or Area', 'Region Name', 'Sub-region Name', 'Intermediate Region Name', 'iso2', 'iso3', 'm49']], left_on='economy', right_on='iso3', how='left')

        # rename and drop duplicate columns
        wb_df.drop(columns=['iso3'], inplace=True)
        wb_df = wb_df.rename(columns={'series': 'Indicator', 'economy': 'iso3'})

        # add indicator description
        indicator_code_description_map = {d['id']: d['value'] for d in query_result}
        wb_df['Indicator Description'] = wb_df['Indicator'].map(indicator_code_description_map)

        # reorder columns
        column_order = ['Indicator', 'Indicator Description', 'Country or Area', 'Region Name', 'Sub-region Name', 'Intermediate Region Name', 'iso2', 'iso3', 'm49'] + [col for col in wb_df.columns if col.startswith('YR')]
        df = wb_df[column_order]

        # create melted df
        df_melted = wb_df.melt(id_vars=['Country or Area', 'Indicator', 'Indicator Description', 'Region Name', 'Sub-region Name', 'Intermediate Region Name', 'iso2', 'iso3', 'm49'], var_name='Year', value_name='Value')
        df_melted['Year'] = df_melted['Year'].str.extract('(\d{4})').astype(int)
        st.session_state.wb_df_melted = df_melted

        st.dataframe(df)
        st.session_state.wb_df = df

    else:
        df = None
        df_melted = None
        st.session_state.wb_df = df
        st.session_state.wb_df_melted = df_melted

except Exception as e:
    st.error(f"no data retrieved. this is most likely due to blank indicator, country, or time values. please ensure there are values for the indicator, country, and time range. \n\n Error: {e}")
    df = None
    df_melted = None
    st.session_state.wb_df = df
    st.session_state.wb_df_melted = df_melted

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")


@st.fragment
def show_plots():
    st.subheader("Explore Data")
    if st.session_state.wb_df_melted is not None:
        
        # plot country indicators
        try:
            fig = px.line(
                st.session_state.wb_df_melted, 
                x='Year', 
                y='Value', 
                color='Country or Area', 
                symbol='Indicator',
                markers=True,
                labels={'Country or Area': 'Country', 'Indicator': 'Indicator', 'Indicator Description': 'Indicator Description', 'Value': 'Value', 'Year': 'Year'},
                title="Time Series of Indicators by Country and Indicator"
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating Time Series Graph:\n\n{e}")

        # region line chart
        with st.spinner("getting region data"):
            region_df = wb.data.DataFrame(selected_indicators, time=list(range(selected_years[0], selected_years[1]))).reset_index()

            if 'series' not in region_df.columns:
                region_df['series'] = selected_indicators[0]

            # add country reference codes
            region_df = region_df.merge(iso3_reference_df[['Region Name', 'Sub-region Name', 'iso3']], left_on='economy', right_on='iso3', how='left')

            # rename and drop duplicate columns
            region_df.drop(columns=['iso3'], inplace=True)
            region_df = region_df.rename(columns={'series': 'Indicator', 'economy': 'iso3'})
            
            region_df_melted = region_df.melt(id_vars=['Indicator', 'Region Name', 'Sub-region Name', 'iso3'], var_name='Year', value_name='Value')
            region_df_melted['Year'] = region_df_melted['Year'].str.extract('(\d{4})').astype(int)
            
            region_avg_df = region_df_melted.groupby(['Indicator', 'Region Name', 'Year'])['Value'].mean().reset_index()
            world_avg_df = region_avg_df.groupby(['Indicator', 'Year'])['Value'].mean().reset_index()

            world_avg_df['Region Name'] = 'World'
            world_avg_df.rename(columns={'Value': 'Value'}, inplace=True)
            region_avg_df = pd.concat([region_avg_df, world_avg_df], ignore_index=True)

        try:
            fig = px.line(
                region_avg_df, 
                x='Year', 
                y='Value', 
                color='Region Name', 
                symbol='Indicator',
                markers=True,
                labels={'Country or Area': 'Country', 'Indicator': 'Indicator', 'Indicator Description': 'Indicator Description', 'Value': 'Value', 'Year': 'Year'},
                title="Time Series of Indicators by Region and Indicator"
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating Time Series Graph:\n\n{e}")


        # map graph
        st.markdown("###### Choose an Indicator to show on the map")
        indicator_descriptions = df_melted['Indicator Description'].unique()
        selected_indicator = st.selectbox("select indicator to show on map:", indicator_descriptions, label_visibility="collapsed")
        indicator_description_code_map = {d['value']: d['id'] for d in query_result}
        selected_code = indicator_description_code_map[selected_indicator]
        indicator_df = df_melted[(df_melted['Indicator'] == selected_code)]

        most_recent_year_with_value = indicator_df.dropna(subset=['Value'])
        most_recent_year = most_recent_year_with_value['Year'].max()
        map_df = most_recent_year_with_value[most_recent_year_with_value['Year'] == most_recent_year]

        try:
            fig = px.choropleth(
                map_df,
                locations='iso3',
                color='Value',
                hover_name='Country or Area',
                color_continuous_scale='Viridis',
                projection='natural earth',
                title="Map of Indicator Value"
            )

            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error generating Map Graph:\n\n{e}")


    else:
        st.write("data not available for the selected indicator(s), countries, and year(s).")
show_plots()


st.markdown("<hr>", unsafe_allow_html=True)
st.write("")


@st.fragment
def show_summary():
    """
    Show Summary statistics on variables.
    """

    st.markdown("### Variable Summary")
    if st.session_state.wb_df is not None and not st.session_state.wb_df.empty:
        if not st.session_state.wb_df.empty:
            summary = st.session_state.wb_df.describe()

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


@st.fragment
def show_chatbot():
    st.subheader("Natural Language Analysis")
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

            if st.session_state.wb_df is not None:
                # df_string = summarize_dataframe(st.session_state.wb_df)
                df_string = st.session_state.wb_df.to_string()
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
show_chatbot()


st.markdown("<hr>", unsafe_allow_html=True)
st.write("")


@st.fragment
def show_mitosheet():
    st.subheader("Mitosheet Spreadsheet")
    if st.session_state.wb_df_melted is not None and not st.session_state.wb_df_melted.empty:
        new_dfs, code = spreadsheet(st.session_state.wb_df_melted)
        if code:
            st.markdown("##### Generated Code:")
            st.code(code)
    else:
        st.write("data not available for the selected indicator(s), countries, and year(s).")
show_mitosheet()


st.markdown("<hr>", unsafe_allow_html=True)
st.write("")


@st.fragment
def show_pygwalker():
    st.subheader("PyGWalker Data Visualization Tool")
    if st.session_state.wb_df_melted is not None and not st.session_state.wb_df_melted.empty:
        init_streamlit_comm()
        @st.cache_resource
        def get_pyg_html(df: pd.DataFrame) -> str:
            html = get_streamlit_html(st.session_state.wb_df_melted, spec="./gw0.json", use_kernel_calc=True, debug=False)
            return html

        components.html(get_pyg_html(st.session_state.wb_df_melted), width=1300, height=1000, scrolling=True)
    else:
        st.write("data not available for the selected indicator(s), countries, and year(s).")
show_pygwalker()