import streamlit as st
import wbgapi as wb
import pandas as pd
import plotly.express as px

# cached functions for retreiving metadata
@st.cache_data
def get_databases():
    return [(database["id"], database["name"]) for database in wb.source.list()]

@st.cache_data
def get_query_result(search_query, db):
    return list(wb.series.list(q=search_query))

@st.cache_data
def get_countries():
    return list(wb.economy.list())
    


# title and introduction
st.title("OSAA SMU's World Bank Data Dashboard")

st.markdown("The WorldBank Data Dashboard allows for exploratory data analysis of the World Bank's Data. First, select one of the WorldBank's databases to use. Then select the indicators, countries, and time range to get data for. Indicators can be filtered with keywords. For example, if you are using the *Doing Business* database and interested in indicators related to construction, enter *construction* into the search box to limit the indicators to only those that mention construction.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

# read in iso3 code reference df
iso3_reference_df = pd.read_csv('iso3_country_reference.csv')

st.markdown("#### Select Database:")
databases = get_databases()
selected_db_name = st.selectbox("available indicators:", [database[1] for database in databases], label_visibility="collapsed")
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

formatted_indicators = [f"{indicator['id']} - {indicator['value']}" for indicator in query_result]
selected_indicator_names = st.multiselect("available indicators:", formatted_indicators, label_visibility="collapsed", placeholder="select indicator(s)")
selected_indicators = [indicator.split(' - ')[0] for indicator in selected_indicator_names]

st.markdown("#### Select Countries:")
countries = get_countries()
formatted_countries = [f"{country['id']} - {country['value']}" for country in countries]
selected_countries = st.multiselect("available countries:", formatted_countries, label_visibility="collapsed", placeholder="select countries")
selected_iso3_codes = [entry.split(' - ')[0] for entry in selected_countries]


st.markdown("#### Select Time Range")
time_selection = st.radio("Which time selection method", ["Time Range", "Most Recent Value"], label_visibility="collapsed")
if time_selection == "Time Range":
    selected_years = st.slider( "Select a range of years:", min_value=1960, max_value=2024, value=(1960, 2024), step=1, label_visibility="collapsed")
    try:
        
        # get world bank data
        if st.button("get data", type='primary', use_container_width=True):
            wb_df = wb.data.DataFrame(selected_indicators, selected_iso3_codes, list(range(selected_years[0], selected_years[1]))).reset_index()

            # deal with edge cases where there is no economy column
            if 'economy' not in wb_df.columns:
                wb_df['economy'] = selected_iso3_codes[0]

            # deal with edge cases where there is no series column
            if 'series' not in wb_df.columns:
                wb_df['series'] = selected_indicators[0]
        
            # add country reference codes
            df = wb_df.merge(iso3_reference_df[['Country or Area', 'Region Name', 'Sub-region Name', 'iso2', 'iso3', 'm49']], left_on='economy', right_on='iso3', how='left')
            
            # rename and drop duplicate columns
            df.drop(columns=['iso3'], inplace=True)
            df = df.rename(columns={'series': 'Indicator', 'economy': 'iso3'})

            # reorder columns
            column_order = ['Indicator', 'Country or Area', 'Region Name', 'Sub-region Name', 'iso2', 'iso3', 'm49'] + [col for col in df.columns if col.startswith('YR')]
            df = df[column_order]

            st.dataframe(df)
        else:
            df = None
    except Exception as e:
        st.error(f"no data retrieved. this is most likely due to blank indicator, country, or time values. please ensure there are values for the indicator, country, and time range. \n\n Error: {e}")
        df = None
else:
    mrv = st.number_input("choose the number of years to get the most recent value within", value=5, placeholder="enter a number of years")
    try:
       # get world bank data
        if st.button("get data", type='primary', use_container_width=True):
            wb_df = wb.data.DataFrame(selected_indicators, selected_iso3_codes, mrv=mrv).reset_index()

            # deal with edge cases where there is no economy column
            if 'economy' not in wb_df.columns:
                wb_df['economy'] = selected_iso3_codes[0]

            # deal with edge cases where there is no series column
            if 'series' not in wb_df.columns:
                wb_df['series'] = selected_indicators[0]
        
            # add country reference codes
            df = wb_df.merge(iso3_reference_df[['iso3', 'Country or Area', 'Region Name', 'Sub-region Name', 'iso2', 'm49']], left_on='economy', right_on='iso3', how='left')
            
            # rename and drop duplicate columns
            df.drop(columns=['iso3'], inplace=True)
            df = df.rename(columns={'series': 'Indicator', 'economy': 'iso3'})

            # reorder columns
            column_order = ['Indicator', 'Country or Area', 'Region Name', 'Sub-region Name', 'iso2', 'iso3', 'm49'] + [col for col in df.columns if col.startswith('YR')]
            df = df[column_order]

            st.dataframe(df)
        else:
            df = None
    except Exception as e:
        st.error(f"Error retrieving the data. This is most likely due to blank indicator, country, or time values. Please ensure there are values for the indicator, country, and time range. \n\n Error: {e}")
        df = None

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

st.subheader("Explore Data")
if df is not None:
    df_melted = df.melt(id_vars=['Country or Area', 'Indicator', 'Region Name', 'Sub-region Name', 'iso2', 'iso3', 'm49'], var_name='Year', value_name='Value')
    df_melted['Year'] = df_melted['Year'].str.extract('(\d{4})').astype(int)
    
    try:
        fig = px.line(
            df_melted, 
            x='Year', 
            y='Value', 
            color='Country or Area', 
            symbol='Indicator',
            markers=True,
            labels={'Country or Area': 'Country', 'Indicator': 'Indicator', 'Value': 'Value', 'Year': 'Year'},
            title="Time Series of Indicators by Country and Indicator"
        )

        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error generating Time Series Graph:\n\n{e}")

    try:
        fig = px.choropleth(
            df_melted,
            locations='iso3',
            color='Value',
            hover_name='iso3',
            color_continuous_scale='Viridis',
            projection='natural earth',
            title="Map of Indicator Value"
        )

        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error generating Map Graph:\n\n{e}")

else:
    st.write("data not available for the selected indicator(s), countries, and year(s).")