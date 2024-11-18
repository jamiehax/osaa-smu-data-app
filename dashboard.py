import streamlit as st
import pandas as pd
from components import df_summary, llm_data_analysis, show_mitosheet, show_pygwalker


# page chat id
chat_session_id = 'data-dashboard-chat-id'

# title and introduction
st.title("OSAA SMU's Data Dashboard")
st.markdown("The Data Dashboard allows for exploratory data analysis on a dataset through quick access to summary statistics and natural language conversations with an AI chatbot that has the ability to understand the dataset. First select a dataset to view by searching the available datasets by name or uploading your own. Once you have selected a dataset, you can filter and subset the dataset to only focus on the area(s) of interest. Once you have selected and filtered a dataset, you can view the summary statistics on that data. To generate and download a more detailed summary, go to the *Dataset Profile Report* section once you have selected and filtered the dataset. Use the *Natural Language Queries* section to understand the data by asking natural language questions to a chatbot that understands the data.")
st.markdown("<hr>", unsafe_allow_html=True)
st.write("")


# find and choose a dataset
st.subheader("Upload and Filter a Dataset")
uploaded_df = st.file_uploader("Choose a file", type=["csv", "xlsx"], label_visibility="collapsed")

if uploaded_df is not None:
    df_name = uploaded_df.name
    if uploaded_df.name.endswith('.csv'):
        df = pd.read_csv(uploaded_df)
    elif uploaded_df.name.endswith('.xlsx'):
        df = pd.read_csv(uploaded_df)
else:
    df = None

# filter the dataset
if df is not None:
    with st.container(height=500):
        st.markdown("##### Column Filters")
        
        selected_columns = st.multiselect('select columns to filter:', df.columns.tolist(), df.columns.tolist())
        
        filtered_df = df.copy()
        
        # iterate over columns and display filters for each
        for col in selected_columns:

            st.markdown(f"##### Filter by {col}")

            # numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].isna().all() or df[col].min() == df[col].max():
                    st.markdown(f"Cannot filter *{col}* because it has invalid or constant values.")
                else:
                    # Determine column type and set slider
                    if pd.api.types.is_integer_dtype(df[col]):
                        min_val, max_val = int(df[col].min()), int(df[col].max())
                        selected_range = st.slider(f"Select range for {col} (int):", min_val, max_val, (min_val, max_val), step=1)
                    elif pd.api.types.is_float_dtype(df[col]):
                        min_val, max_val = float(df[col].min()), float(df[col].max())
                        selected_range = st.slider(f"Select range for {col} (float):", min_val, max_val, (min_val, max_val))

                    # Apply filter while retaining NaN values (if needed)
                    filtered_df = filtered_df[
                        (filtered_df[col].isna()) |  # Keep rows with NaN
                        ((filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1]))
                    ]
            
            # categorical columns
            elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                if df[col].isna().all() or df[col].nunique() == 1:
                     st.markdown(f"cannot filter *{col}* because it has invalid or constant values.")
                else:
                    unique_vals = df[col].dropna().unique().tolist()
                    selected_vals = st.multiselect(f"Select values for {col}:", unique_vals, unique_vals)
                    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
            
            # datetime columns
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


   
# if there is a dataset selected, show the dataset and data tools
if filtered_df is not None and not filtered_df.empty:

    # display the dataset
    st.markdown("### Dataset")
    st.write(filtered_df)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("")

    # show summary statistics
    st.markdown("### Variable Summary")
    df_summary(filtered_df)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("") 

    # natural language dataset exploration
    st.subheader("Natural Language Analysis")
    st.write("Use this chat bot to understand the data with natural language questions. Ask questions about the data and the chat bot will provide answers in natural language, as well as code (Python, R, etc.).")
    llm_data_analysis(filtered_df, chat_session_id)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("") 

    # Mitosheet
    st.subheader("Mitosheet Spreadsheet")
    show_mitosheet(filtered_df)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("") 

    # PyGWalker
    st.subheader("PyGWalker Graphing Tool")
    show_pygwalker(filtered_df)
    
elif filtered_df is not None and filtered_df.empty:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("") 
    st.markdown("### Dataset")
    st.write("no data returned for selected filters")
