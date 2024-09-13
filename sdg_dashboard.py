import streamlit as st
import pandas as pd
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


chat_session_id = 'sdg-dashboard-id'

# create session states
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'formatted_chat_history' not in st.session_state:
    st.session_state.formatted_chat_history = {}
if 'sdg_df' not in st.session_state:
    st.session_state['sdg_df'] = None


# base url for SDG requests
BASE_URL = "https://unstats.un.org/sdgs/UNSDGAPIV5"

# read in iso3 code reference df
iso3_reference_df = pd.read_csv('content/iso3_country_reference.csv')
iso3_reference_df['m49'] = iso3_reference_df['m49'].astype(str)

# title and introduction
st.title("OSAA SMU's SDG Data Dashboard")

st.markdown("The SDG Dashboard allows for exploratory data analysis of the United Nations Sustainable Development Goals DataBase. Explore the 17 sustainable development goals and their corresponding indicators, and select and download data by indicator, country, and time range. Create automatic interactive time series graphs on the selected data.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

st.markdown("### Explore Sustainable Development Goals")
goals_url = "v1/sdg/Goal/List?includechildren=false"
goals_data = get_data(f"{BASE_URL}/{goals_url}")
if not isinstance(goals_data, Exception):
    selected_goal_title = st.selectbox("select goal to explore", [f"{goal['code']}. {goal['title']}" for goal in goals_data], label_visibility="collapsed")
    selected_goal_code = next(goal['code'] for goal in goals_data if f"{goal['code']}. {goal['title']}" == selected_goal_title)
    selected_goal_data = next(goal for goal in goals_data if f"{goal['code']}. {goal['title']}" == selected_goal_title)

    st.write(selected_goal_data['description'])

    st.markdown(f"##### Available Indicators for *{selected_goal_title}*")
    indicator_url = "v1/sdg/Indicator/List"
    indicator_data = get_data(f"{BASE_URL}/{indicator_url}")
    if not isinstance(indicator_data, Exception):
        for i, indicator in enumerate(indicator_data):           
            if indicator['goal'] == selected_goal_code:
                series_codes = [series_entry['code'] for series_entry in indicator['series']]
                st.markdown(f"**{indicator['code']}:** {indicator['description']}.")


st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

st.markdown("#### Select Indicators")
indicators = [f"{indicator['code']}: {indicator['description']}" for indicator in indicator_data]
if indicator_data is not None:
    selected_indicator_names = st.multiselect("select indicators", indicators, label_visibility="collapsed")
    selected_indicator_codes = [entry.split(': ')[0] for entry in selected_indicator_names]

st.markdown("#### Select Countries")
country_code_url = "v1/sdg/GeoArea/List"
country_code_data = get_data(f"{BASE_URL}/{country_code_url}")

# if not isinstance(country_code_data, Exception):
#     selected_country_code_names = st.multiselect("select countries", [f"{country_code['geoAreaCode']} - {country_code['geoAreaName']}" for country_code in country_code_data], label_visibility="collapsed")
#     selected_country_codes = [entry.split(' - ')[0] for entry in selected_country_code_names]
#     st.write(selected_country_codes)

if not isinstance(country_code_data, Exception):

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

    def get_countries_by_region(region):
        return iso3_reference_df[iso3_reference_df['Region Name'] == region]['m49'].tolist()

    selected_countries = []
    for region in selected_regions:
        selected_countries.extend(get_countries_by_region(region))

    # remove duplicates
    selected_countries = list(set(selected_countries))
    selected_country_names = iso3_reference_df[iso3_reference_df['m49'].isin(selected_countries)]['Country or Area'].tolist()
    m49_to_name = dict(zip(iso3_reference_df['m49'], iso3_reference_df['Country or Area']))
    selected_countries_formatted = [f"{country_code} - {m49_to_name[country_code]}" for country_code in selected_countries]

    available_countries = list(zip(iso3_reference_df['m49'].tolist(), iso3_reference_df['Country or Area'].tolist()))
    available_countries_formatted = [f"{country[0]} - {country[1]}" for country in available_countries]

    selected_countries = st.multiselect(
        "Available countries:",
        available_countries_formatted,
        default=selected_countries_formatted,
        label_visibility="collapsed",
        placeholder="select by country"
    )

    selected_country_codes = [entry.split(' - ')[0] for entry in selected_countries]


st.markdown("#### Select Time Range")
selected_years = st.slider( "Select a range of years:", min_value=1963, max_value=2023, value=(1963, 2023), step=1, label_visibility="collapsed")

# get data
indicator_params = "indicator=" + "&indicator=".join(selected_indicator_codes)
country_params = "&areaCode=" + "&areaCode=".join(selected_country_codes)
year_params = "&timePeriod=" + "&timePeriod=".join([str(i) for i in range(selected_years[0], selected_years[1] + 1)])
page_size = 100
data_url = f"{BASE_URL}/v1/sdg/Indicator/Data?{indicator_params}{country_params}{year_params}&pageSize={page_size}"

st.write("NOTE: the maximum number of pages defaults to 1000. Each page contains 100 rows of data. If you need more than 10,000 rows, increase the maximum page size accordingly. Very large queries may result in app timeouts.")

col1, col2 = st.columns(2)
with col1:
    max_pages = st.number_input("Insert a number", min_value=1, value=None, placeholder="maximum number of pages (defaults to 1000)", label_visibility="collapsed")
    if max_pages is None:
        max_pages = 1000

with col2:
    if st.button("get data", type='primary', use_container_width=True):

        # loop over pages to get all data
        extracted_data = []
        page_num = 1
        for page_num in range(1, max_pages + 1):
            data = get_data(f'{data_url}&page={page_num}')

            # break if no data
            if not data.get('data', []):
                break

            if not isinstance(data, Exception):
                if len(data['data']) < 1:
                    st.write("no data returned for the selected countries, indicators, and years.")
                else:
                    for entry in data["data"]:
                        extracted_data.append({
                            "Indicator": entry["indicator"][0],
                            "Value": entry["value"],
                            "Year": entry["timePeriodStart"],
                            "m49": entry["geoAreaCode"],
                            "Country": entry["geoAreaName"],
                            "Series": entry["series"],
                            "Series Description": entry["seriesDescription"]
                        })

            else:
                st.error(f"An error occurred while getting the data: \n\n {data}.")

            if len(data.get('data', [])) < page_size:
                break

        df = pd.DataFrame(extracted_data)

        if not df.empty:

            # add country reference codes
            df = df.merge(iso3_reference_df[['Country or Area', 'Region Name', 'Sub-region Name', 'iso2', 'iso3', 'm49']], left_on='m49', right_on='m49', how='left')

            # clean dataframe
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            df['Year'] = df['Year'].astype(int)

            # reorder columns
            column_order = ['Indicator', 'Series', 'Year', 'Country or Area', 'Value', 'Region Name', 'Sub-region Name', 'iso2', 'iso3', 'm49', 'Series Description'] + [col for col in df.columns if col.startswith('YR')]
            df = df[column_order]

            st.session_state.sdg_df = df

        else:
            df = None
            st.session_state.sdg_df = df

    else:
        df = None
        st.session_state.sdg_df = df


if st.session_state.sdg_df is not None:
    if not st.session_state.sdg_df.empty:
        st.dataframe(st.session_state.sdg_df)
    else:
        st.write("no data available for the selection")



st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

@st.fragment
def show_plots():
    st.subheader("Explore Data")
    if st.session_state.sdg_df is not None:
        try:
            fig = px.line(
                st.session_state.sdg_df, 
                x='Year', 
                y='Value', 
                color='Country or Area', 
                symbol='Series',
                markers=True,
                labels={'Country or Area': 'Country', 'Series': 'Series', 'Series Description': 'Series Description', 'Value': 'Value', 'Year': 'Year'},
                title="Time Series of Indicators by Country and Indicator"
            )

            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error generating graph:\n\n{e}")


        try:           
            st.markdown("###### Choose an Series to show on the map")
            series_descriptions = st.session_state.sdg_df['Series Description'].unique()
            selected_series= st.selectbox("select indicator to show on map:", series_descriptions, label_visibility="collapsed")
            series_df = st.session_state.sdg_df[(st.session_state.sdg_df['Series Description'] == selected_series)]

            most_recent_year_with_value = series_df.dropna(subset=['Value'])
            most_recent_year = most_recent_year_with_value['Year'].max()
            map_df = most_recent_year_with_value[most_recent_year_with_value['Year'] == most_recent_year]

            map_df = series_df[series_df['Year'] == most_recent_year]

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

            if st.session_state.sdg_df is not None:
                df_string = summarize_dataframe(st.session_state.sdg_df)
                df_string = st.session_state.sdg_df.to_string()
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
    if st.session_state.sdg_df is not None and not st.session_state.sdg_df.empty:
        new_dfs, code = spreadsheet(st.session_state.sdg_df)
        if code:
            st.markdown("##### Generated Code:")
            st.code(code, language='python')
    else:
        st.write("data not available for the selected indicator(s), countries, and year(s).")
show_mitosheet()

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

@st.fragment
def show_pygwalker():
    st.subheader("PyGWalker Data Visualization Tool")
    if st.session_state.sdg_df is not None and not st.session_state.sdg_df.empty:
        init_streamlit_comm()
        @st.cache_resource
        def get_pyg_html(df: pd.DataFrame) -> str:
            html = get_streamlit_html(st.session_state.sdg_df, spec="./gw0.json", use_kernel_calc=True, debug=False)
            return html

        components.html(get_pyg_html(st.session_state.sdg_df), width=1300, height=1000, scrolling=True)
    else:
        st.write("no dataset selected or the selected filters have resulted in an empty dataset.")
show_pygwalker()