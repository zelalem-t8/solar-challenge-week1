import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
@st.cache_data
def load_data():
    countries = ['sierraleone', 'togo','benin']
    dfs = []
    for country in countries:
        df = pd.read_csv(f'./data/{country}_clean.csv')
        df['Country'] = country.capitalize()
        # Convert Timestamp column to datetime
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        dfs.append(df)
    return pd.concat(dfs)

df = load_data()

# Sidebar filters
st.sidebar.header('Filters')

# Single country selection
selected_country = st.sidebar.selectbox(
    'Select country',
    df['Country'].unique()
)

selected_metric = st.sidebar.selectbox(
    'Select solar metric',
    ['GHI', 'DNI', 'DHI']
)

# Filter data
filtered = df[df['Country'] == selected_country]

# Main content
st.title('Solar Farm Analysis Dashboard')

# Histogram for all data
st.header('Histogram for All Data')
fig_all_hist, ax_all_hist = plt.subplots(figsize=(10,6))
sns.histplot(data=df, x=selected_metric, hue='Country', kde=True, alpha=0.5, ax=ax_all_hist)
st.pyplot(fig_all_hist)

# Correlation matrix for all data
st.header('Correlation Matrix for All Data')
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

fig_corr, ax_corr = plt.subplots(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax_corr)
st.pyplot(fig_corr)

# Selected country analysis
st.header(f'Analysis for {selected_country}')

# Metric visualization
tab1, tab2 = st.tabs(["Box Plot", "Scatter Plot"])

with tab1:
    fig_box, ax_box = plt.subplots(figsize=(10,6))
    sns.boxplot(data=filtered, y=selected_metric, ax=ax_box)
    st.pyplot(fig_box)

with tab2:
    st.subheader("Scatter Plot")
    available_columns = [col for col in filtered.columns if col not in ['Country', 'Timestamp'] and pd.api.types.is_numeric_dtype(filtered[col])]
    x_axis = st.selectbox("Select X-axis variable", available_columns)
    y_axis = st.selectbox("Select Y-axis variable", available_columns, index=1 if len(available_columns) > 1 else 0)
    
    fig_scatter, ax_scatter = plt.subplots(figsize=(10,6))
    sns.scatterplot(data=filtered, x=x_axis, y=y_axis, alpha=0.6, ax=ax_scatter)
    st.pyplot(fig_scatter)

# Time series for selected country
if 'Timestamp' in filtered.columns:
    st.header('Time Series Analysis')
    
    # Ensure Timestamp is datetime and set it as index
    filtered = filtered.copy()
    filtered['Timestamp'] = pd.to_datetime(filtered['Timestamp'])
    filtered.set_index('Timestamp', inplace=True)
    
    # Resample only numeric columns
    numeric_filtered = filtered.select_dtypes(include=[np.number])
    daily = numeric_filtered.resample('D').mean()

    # Plot the selected metric
    if selected_metric in daily.columns:
        fig_ts, ax_ts = plt.subplots(figsize=(12,4))
        daily[selected_metric].plot(ax=ax_ts)
        ax_ts.set_title(f'Daily Average of {selected_metric} - {selected_country}')
        st.pyplot(fig_ts)
    else:
        st.warning(f"{selected_metric} not available in the resampled data.")
else:
    st.warning("Time series analysis disabled - no Timestamp column found")

# Summary table
st.header('Summary Statistics')
summary = filtered[selected_metric].describe().to_frame().T
st.dataframe(summary.style.background_gradient(cmap='Blues'))