import streamlit as st
from preprocess import preprocess
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


# get dataframes
df_paths = {
    "Education Data": "data/education_test.csv",
    "Income Data": "data/income_test.csv",
    "Poverty Data": "data/poverty_test.csv"
}
dataframes = {name: preprocess(path) for name, path in df_paths.items()}


# select desired dataframe
st.sidebar.header("Select Dataset")
selected_df_name = st.sidebar.selectbox('Select Dataset', list(dataframes.keys()), label_visibility="collapsed")
selected_df = dataframes[selected_df_name]


# filter the dataset
st.header("Dataset Filters")

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

selected_columns = st.multiselect('select columns:', selected_df.columns.tolist(), selected_df.columns.tolist())


# summary section
with st.expander("show a summary of the filtered dataset"):

    summary = selected_df[selected_columns].describe()

    columns = summary.columns
    tabs = st.tabs(columns.to_list())

    for i, column in enumerate(columns):
        with tabs[i]:
            st.subheader(f"summary of: {column}")
            st.markdown(f"**Count**: {summary.loc['count', column]}")
            st.markdown(f"**Mean**: {summary.loc['mean', column]:.2f}")
            st.markdown(f"**Standard Deviation**: {summary.loc['std', column]:.2f}")
            st.markdown(f"**Min**: {summary.loc['min', column]}")
            st.markdown(f"**25th Percentile**: {summary.loc['25%', column]}")
            st.markdown(f"**50th Percentile (Median)**: {summary.loc['50%', column]}")
            st.markdown(f"**75th Percentile**: {summary.loc['75%', column]}")
            st.markdown(f"**Max**: {summary.loc['max', column]}")


# create the dataframe profile and display it
st.header("Profile Report")
if st.button('Generate Profile Report'):
    if selected_df[selected_columns].empty:
        st.write("no data available for the subsetted data.")
    else:
        profile = ProfileReport(selected_df[selected_columns], title=f"Profile Report for {selected_df_name}", explorative=True)
        st_profile_report(profile)
