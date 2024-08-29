import streamlit as st
import wbgapi as wb
import pandas as pd
import plotly.express as px


# cached functions for retreiving metadata
@st.cache_data
def get_databases():
    return [(database["id"], database["name"]) for database in wb.source.list()]

@st.cache_data
def get_indicators():
    return list(wb.series.list())

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

formatted_indicators = [f"{indicator['id']} - {indicator['value']}" for indicator in query_result]
selected_indicator_names = st.multiselect("available indicators:", formatted_indicators, label_visibility="collapsed", placeholder="select indicator(s)")
selected_indicators = [indicator.split(' - ')[0] for indicator in selected_indicator_names]

st.markdown("#### Select Countries:")

countries = get_countries()
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
    return iso3_reference_df[iso3_reference_df['Region Name'] == region]['iso3'].tolist()

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
        df = wb_df.merge(iso3_reference_df[['Country or Area', 'Region Name', 'Sub-region Name', 'iso2', 'iso3', 'm49']], left_on='economy', right_on='iso3', how='left')
        
        # rename and drop duplicate columns
        df.drop(columns=['iso3'], inplace=True)
        df = df.rename(columns={'series': 'Indicator', 'economy': 'iso3'})

        # add indicator description
        indicator_code_description_map = {d['id']: d['value'] for d in query_result}
        df['Indicator Description'] = df['Indicator'].map(indicator_code_description_map)

        # reorder columns
        column_order = ['Indicator', 'Indicator Description', 'Country or Area', 'Region Name', 'Sub-region Name', 'iso2', 'iso3', 'm49'] + [col for col in df.columns if col.startswith('YR')]
        df = df[column_order]

        st.dataframe(df)
    else:
        df = None
except Exception as e:
    st.error(f"no data retrieved. this is most likely due to blank indicator, country, or time values. please ensure there are values for the indicator, country, and time range. \n\n Error: {e}")
    df = None

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

st.subheader("Explore Data")
if df is not None:
    df_melted = df.melt(id_vars=['Country or Area', 'Indicator', 'Indicator Description', 'Region Name', 'Sub-region Name', 'iso2', 'iso3', 'm49'], var_name='Year', value_name='Value')
    df_melted['Year'] = df_melted['Year'].str.extract('(\d{4})').astype(int)
    
    try:
        fig = px.line(
            df_melted, 
            x='Year', 
            y='Value', 
            color='Country or Area', 
            symbol='Indicator Description',
            markers=True,
            labels={'Country or Area': 'Country', 'Indicator Description': 'Indicator Description', 'Value': 'Value', 'Year': 'Year'},
            title="Time Series of Indicators by Country and Indicator"
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating Time Series Graph:\n\n{e}")

    @st.fragment
    def show_map():
        try:
            indicator_descriptions = df_melted['Indicator Description'].unique()
            most_recent_year = df_melted['Year'].max()
            df_melted_mry = df_melted[df_melted['Year'] == most_recent_year]
            
            st.markdown("###### Choose an Indicator to show on the map")
            selected_indicator = st.selectbox("select indicator to show on map:", indicator_descriptions, label_visibility="collapsed")
            indicator_description_code_map = {d['value']: d['id'] for d in query_result}
            selected_code = indicator_description_code_map[selected_indicator]
            map_df = df_melted_mry[(df_melted_mry['Indicator'] == selected_code)]

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

    show_map()

else:
    st.write("data not available for the selected indicator(s), countries, and year(s).")