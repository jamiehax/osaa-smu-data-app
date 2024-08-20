import streamlit as st
import wbgapi as wb
import pandas as pd
import plotly.express as px


# title and introduction
st.title("OSAA SMU's World Bank Data Dashboard")

st.markdown("Explore the World Bank's data.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

st.markdown("### Explore Indicators:")
st.markdown("Explore available indicators by entering keywords. For example, to find indicators related to fossil fuels, enter *fossil fuels*.")
text_col, send_col = st.columns(2)
with text_col:
    search_query = st.text_input(
            "enter your query",
            label_visibility='collapsed',
            placeholder="enter your query"
        )
with send_col:
    query_result = None
    if st.button('search'):
        if search_query:
            query_result = wb.search(search_query)

with st.expander("show results"):
    if query_result:
        st.write(query_result)
    else:
        st.write("please enter a query")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

# find and choose a dataset
st.markdown("### Select Data ")
st.write("Search World Bank Indicators.")

st.markdown("##### Select Indicators:")
indicators = wb.series.list()
selected_indicators = st.multiselect("available indicators:", indicators, label_visibility="collapsed")

st.markdown("##### Select Countries:")
countries = wb.economy.list()
selected_countries = st.multiselect("available countries:", countries, label_visibility="collapsed")

st.markdown("##### Select Time Range:")
selected_years = st.slider( "Select a range of years:", min_value=1960, max_value=2024, value=(1960, 2024), step=1)

try:
    df = wb.data.DataFrame(selected_indicators, selected_countries, list(range(selected_years[0], selected_years[1]))).reset_index()
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
    st.write("data not available for the selected indicator(s) countries, and year(s).")
    filtered_df = None

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

st.subheader("Explore Data")
if filtered_df is not None:
    try:
        df_melted = df.melt(id_vars=['economy', 'series'], var_name='Year', value_name='Value')
        df_melted['Year'] = df_melted['Year'].str.extract('(\d{4})').astype(int)

        fig = px.line(df_melted, 
                  x='Year', 
                  y='Value', 
                  color='economy', 
                  symbol='series',  # Different markers for different series
                  markers=True,
                  labels={'economy': 'Country', 'series': 'Indicator', 'Value': 'Value', 'Year': 'Year'},
                  title="Time Series of Indicators by Country and Indicator")
    
        st.plotly_chart(fig)
    except Exception as e:
        st.error("Error generating graph. This is likely due to the data having only one selected country, year, and or indicator. That graph is not supported yet. Please make sure the selected data has more than 1 country, indicator, and year.")

else:
    st.write("data not available for the selected indicator(s) countries, and year(s).")
