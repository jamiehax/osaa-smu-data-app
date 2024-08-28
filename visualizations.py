import streamlit as st
import streamlit.components.v1 as components
from helper_functions import get_dataset_names, get_df
import pandas as pd
from pygwalker.api.streamlit import init_streamlit_comm, get_streamlit_html


# title and introduction
st.title("OSAA SMU's Data Visulization Tool")

st.markdown("The Data Visualization tool uses *PyGWalker* to automatically create data visualizations like graphs and charts on uploaded datasets. First, select a dataset by choosing from one of the existing datasets or uploading your own. Then, filter the dataset as needed (removing columns, filtering data by column values, etc.). The selected and filtered dataset can then be used to create visualizations with PyGWalker.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

# find and choose a dataset
st.subheader("Select a Dataset")
st.write("Either search through existing datasets or upload your own dataset as a CSV or Excel file.")


st.markdown("##### Search Datasets")
dataset_names = get_dataset_names(st.session_state.db_path)
df_name = st.selectbox("find a dataset", dataset_names, index=None, placeholder="search datasets...", label_visibility="collapsed")

if df_name is not None:
    df = get_df(st.session_state.db_path, df_name)
else:
    df = None
    st.session_state.report = None


st.markdown("##### Upload a Dataset (CSV or excel)")
uploaded_df = st.file_uploader("Choose a file", type=["csv", "xlsx"], label_visibility="collapsed")

if uploaded_df is not None:
    df_name = uploaded_df.name
    if uploaded_df.name.endswith('.csv'):
        df = pd.read_csv(uploaded_df)
    elif uploaded_df.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_df)


if df is not None: st.write(df)
st.write("")

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
                    if pd.api.types.is_integer_dtype(df[col]):
                        min_val, max_val = int(df[col].min()), int(df[col].max())
                        selected_range = st.slider(f"Select range for {col} (int):", min_val, max_val, (min_val, max_val), step=1)
                    elif pd.api.types.is_float_dtype(df[col]):
                        min_val, max_val = float(df[col].min()), float(df[col].max())
                        selected_range = st.slider(f"Select range for {col} (float):", min_val, max_val, (min_val, max_val))
            
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

            # download
            if df is not None:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="download filtered data as a CSV file",
                    data=csv,
                    file_name='data.csv',
                    mime='text/csv',
                    disabled=(df is None),
                    type='primary',
                    use_container_width=True
                )

else:
    st.write("No dataset selected")
    filtered_df = None

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")


st.subheader("Data Visualization Tool")
if filtered_df is not None:
    init_streamlit_comm()
    @st.cache_resource
    def get_pyg_html(df: pd.DataFrame) -> str:
        html = get_streamlit_html(df, spec="./gw0.json", use_kernel_calc=True, debug=False)
        return html

    components.html(get_pyg_html(df), width=1300, height=1000, scrolling=True)
else:
    st.write("no dataset selected")