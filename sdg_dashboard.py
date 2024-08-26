import streamlit as st
import pandas as pd
import plotly.express as px
import requests


@st.cache_data
def get_data(url):
    """
    Function to get data from the passed URL through an HTTPS request and return it as a JSON object. Data is cached so that function does not rerun when URL doesn't change.
    """
    try:
        data = requests.get(url).json()
    except Exception as e:
        data = f"Error getting data: {e}"

    return data


# base url for SDG requests
BASE_URL = "https://unstats.un.org/sdgs/UNSDGAPIV5"

# title and introduction
st.title("OSAA SMU's SDG Data Dashboard")

st.markdown("Explore the United Nations Sustainable Development Groups DataBase.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

st.markdown("### Explore Sustainable Development Goals and Indicators")
goals_url = "v1/sdg/Goal/List?includechildren=false"
goals_data = get_data(f"{BASE_URL}/{goals_url}")
selected_goal_title = st.selectbox("select goal to explore", [f"{goal['code']}. {goal['title']}" for goal in goals_data], label_visibility="collapsed")
selected_goal_code = next(goal['code'] for goal in goals_data if f"{goal['code']}. {goal['title']}" == selected_goal_title)
selected_goal_data = next(goal for goal in goals_data if f"{goal['code']}. {goal['title']}" == selected_goal_title)

st.markdown(f"##### {selected_goal_title}")
st.write(selected_goal_data['description'])

st.markdown("##### Available Indicators")
indicator_url = "v1/sdg/Indicator/List"
indicator_data = get_data(f"{BASE_URL}/{indicator_url}")
with st.expander("show indicators"):
    for i, indicator in enumerate(indicator_data):           
        if indicator['goal'] == selected_goal_code:
            series_codes = [series_entry['code'] for series_entry in indicator['series']]
            st.markdown(f"**{indicator['code']}:** {indicator['description']}.")


st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

st.markdown("### Select Data")

st.markdown("##### Select Indicators")
indicators = [f"{indicator['code']}: {indicator['description']}" for indicator in indicator_data]
selected_indicator_names = st.multiselect("select indicators", indicators, label_visibility="collapsed")
selected_indicator_codes = [entry.split(': ')[0] for entry in selected_indicator_names]

st.markdown("##### Select Countries")
country_code_url = "v1/sdg/GeoArea/List"
country_code_data = get_data(f"{BASE_URL}/{country_code_url}")
selected_country_code_names = st.multiselect("select countries", [f"{country_code['geoAreaCode']} - {country_code['geoAreaName']}" for country_code in country_code_data], label_visibility="collapsed")
selected_country_codes = [entry.split(' - ')[0] for entry in selected_country_code_names]

st.markdown("##### Select Time Range")
selected_years = st.slider( "Select a range of years:", min_value=1963, max_value=2023, value=(1963, 2023), step=1, label_visibility="collapsed")

# get data
indicator_params = "indicator=" + "&indicator=".join(selected_indicator_codes)
country_params = "&areaCode=" + "&areaCode=".join(selected_country_codes)
year_params = "&timePeriod=" + "&timePeriod=".join([str(i) for i in range(selected_years[0], selected_years[1] + 1)])
data_url = f"{BASE_URL}/v1/sdg/Indicator/Data?{indicator_params}{country_params}{year_params}"

if st.button("get data"):
    data = get_data(data_url)
    extracted_data = []
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
    df = pd.DataFrame(extracted_data)
else:
    df = None


if df is not None:
    if not df.empty:
        st.write(df)
    else:
        st.write("no data available for the selection")

