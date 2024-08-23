import streamlit as st
import wbgapi as wb
import pandas as pd
import plotly.express as px

# set world bank database
wb.db = 1


# title and introduction
st.title("OSAA SMU's World Bank Data Dashboard")

st.markdown("Explore the World Bank's data.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

# read in iso3 code reference df
iso3_reference_df = pd.read_csv('iso3_country_reference.csv')

st.markdown("#### Select Database:")
databases = [(database["id"], database["name"]) for database in wb.source.list()]
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
query_result = wb.series.list(q=search_query)

formatted_indicators = [f"{indicator['id']} - {indicator['value']}" for indicator in query_result]
selected_indicator_names = st.multiselect("available indicators:", formatted_indicators, label_visibility="collapsed")
selected_indicators = [indicator.split(' - ')[0] for indicator in selected_indicator_names]

st.markdown("#### Select Countries:")
countries = wb.economy.list()
formatted_countries = [f"{country['id']} - {country['value']}" for country in countries]
selected_countries = st.multiselect("available countries:", formatted_countries, label_visibility="collapsed")
selected_iso3_codes = [entry.split(' - ')[0] for entry in selected_countries]


st.markdown("#### Select Time Range")
time_selection = st.radio("Which time selection method", ["Time Range", "Most Recent Value"], label_visibility="collapsed")
if time_selection == "Time Range":
    selected_years = st.slider( "Select a range of years:", min_value=1960, max_value=2024, value=(1960, 2024), step=1, label_visibility="collapsed")

    try:
        # get world bank data
        wb_df = wb.data.DataFrame(selected_indicators, selected_iso3_codes, list(range(selected_years[0], selected_years[1]))).reset_index()
        
        # add country reference codes
        df = wb_df.merge(iso3_reference_df[['Country or Area', 'Region Name', 'Sub-region Name', 'iso2', 'iso3', 'm49']], left_on='economy', right_on='iso3', how='left')
        
        # rename and drop duplicate columns
        df.drop(columns=['iso3'], inplace=True)
        df = df.rename(columns={'series': 'Indicator', 'economy': 'iso3'})

        # reorder columns
        column_order = ['Indicator', 'Country or Area', 'Region Name', 'Sub-region Name', 'iso2', 'iso3', 'm49'] + [col for col in df.columns if col.startswith('YR')]
        df = df[column_order]

        st.write(df)
    except Exception as e:
        df = None
else:
    mrv = st.number_input("choose the number of years to get the most recent value within", value=5, placeholder="enter a number of years")
    try:
        # get world bank data
        wb_df = wb.data.DataFrame(selected_indicators, selected_iso3_codes, mrv=mrv).reset_index()
        
        # add country reference codes
        df = wb_df.merge(iso3_reference_df[['iso3', 'Country or Area', 'Region Name', 'Sub-region Name', 'iso2', 'm49']],
                                left_on='economy', right_on='iso3', how='left')
        
        # reorder columns
        df.drop(columns=['economy'], inplace=True)
        df = df.rename(columns={'series': 'Indicator'})
        columns_to_insert = ['Country or Area', 'Region Name', 'Sub-region Name', 'iso2', 'iso3', 'm49']
        for i, column in enumerate(columns_to_insert):
            df.insert(1 + i, column, df.pop(column))

        st.write(df)
    except Exception as e:
        df = None
            

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
    st.write("data not available for the selected indicator(s), countries, and year(s).")
    filtered_df = None

            
# download
if filtered_df is not None:
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Dataset as a CSV File",
        data=csv,
        file_name='data.csv',
        mime='text/csv',
        disabled=(filtered_df is None)
    )

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

st.subheader("Explore Data")
if filtered_df is not None:
    try:
        df_melted = filtered_df.melt(id_vars=['Country or Area', 'Indicator', 'Region Name', 'Sub-region Name', 'iso2', 'iso3', 'm49'], var_name='Year', value_name='Value')
        df_melted['Year'] = df_melted['Year'].str.extract('(\d{4})').astype(int)

        fig = px.line(df_melted, 
                    x='Year', 
                    y='Value', 
                    color='Country or Area', 
                    symbol='Indicator',
                    markers=True,
                    labels={'Country or Area': 'Country', 'Indicator': 'Indicator', 'Value': 'Value', 'Year': 'Year'},
                    title="Time Series of Indicators by Country and Indicator")

        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error generating graph:\n\n{e}  \n\n This is likely due to the data having only one selected country, year, and or indicator. That graph is not supported yet. Please make sure the selected data has more than 1 country, indicator, and year.")

else:
    st.write("data not available for the selected indicator(s), countries, and year(s).")
