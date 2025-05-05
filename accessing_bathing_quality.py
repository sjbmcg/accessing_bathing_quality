import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# Set page configuration
st.set_page_config(
    page_title="Water Quality Dashboard",
    page_icon="💧",
    layout="wide"
)

# Header
st.title('Water Quality Analysis Dashboard')
st.write("Environmental Agency - Water Quality Monitoring System")

# ---------- DATA LOADING AND PROCESSING FUNCTIONS ----------
# These functions handle data loading and transformation to support analysis

@st.cache_data
def load_data():
    """
    Load and process the water quality data.
    Uses caching to avoid reprocessing on every interaction.
    """
    try:
        # Try to load processed data first
        return pd.read_csv("Bacteria.csv")
    except FileNotFoundError:
        # If not found, process the raw data
        xls = pd.ExcelFile('Anonymised data 2023 mapped.xlsx')
        
        # Create dataset for each excel sheet - Taken from original notebook
        Principle_Data = pd.read_excel(xls, 0)
        Investigation_samples = pd.read_excel(xls, 1)
        Tide_data = pd.read_excel(xls, 3)
        STW_flow_UV = pd.read_excel(xls, 4)
        UV_Wind = pd.read_excel(xls, 7)
        
        # Create datetime columns - Taken from original notebook
        Principle_Data['datetime'] = pd.to_datetime(Principle_Data['Date'].dt.strftime('%Y-%m-%d') + ' ' + Principle_Data['Time GMT'])
        Investigation_samples['datetime'] = pd.to_datetime(Investigation_samples['Date'].dt.strftime('%Y-%m-%d') + ' ' + Investigation_samples['Time GMT'])
        
        # Process data - Column renaming taken from original notebook
        Bacteria_with_rain = calculate_lagged_rainfall(Principle_Data, Investigation_samples)
        Bacteria_with_rain.rename(columns={'Date': 'Date_NA'}, inplace=True)
        Bacteria_with_rain.rename(columns={'datetime': 'Date'}, inplace=True)
        
        # Add tide data - Taken from original notebook
        Tide_data2 = Tide_data[['Date', 'Tide Astronomical (MaOD)']]
        Bacteria = Bacteria_with_rain.merge(Tide_data2, on='Date', how='left')
        
        # Add sewage discharge data - Taken from original notebook
        Bacteria = calculate_lagged_sewage(STW_flow_UV, Bacteria)
        
        # Add UV index data - Taken from original notebook
        Bacteria = calculate_lagged_UV(UV_Wind, Bacteria)
        
        # Add wind speed data - Taken from original notebook
        Bacteria = calculate_lagged_Wind(UV_Wind, Bacteria)
        
        # Save processed data
        Bacteria.to_csv("Bacteria.csv", index=False)
        
        return Bacteria

# Function to calculate cumulative rainfall over different time periods - Taken directly from original notebook
def calculate_lagged_rainfall(rainfall, bacteria):
    """Calculate cumulative rainfall in 24h, 48h, and 72h before each bacteria sample."""
    lag_hours = [24, 48, 72]
    results = []

    for i, row in bacteria.iterrows():
        sample_time = row['datetime']
        new_row = row.copy()

        for h in lag_hours:
            start = sample_time - pd.Timedelta(hours=h)
            end = sample_time
            total_rain = rainfall[(rainfall['datetime'] >= start) & (rainfall['datetime'] < end)]['RF mm (GMT)'].sum()
            new_row[f'Rainfall_last_{h}h'] = total_rain

        results.append(new_row)

    return pd.DataFrame(results)

# Function to calculate average sewage discharge over different time periods - Taken directly from original notebook
def calculate_lagged_sewage(sewage, bacteria):
    """Calculate average sewage discharge in 24h, 48h, and 72h before each bacteria sample."""
    lag_hours = [24, 48, 72]
    results = []

    for i, row in bacteria.iterrows():
        sample_time = row['Date']
        new_row = row.copy()

        for h in lag_hours:
            start = sample_time - pd.Timedelta(hours=h)
            end = sample_time
            mean_flow = sewage[(sewage['Date'] >= start) & (sewage['Date'] < end)]['Flow (l/s)'].mean()
            new_row[f'Average Discharge_last_{h}h'] = mean_flow

        results.append(new_row)

    return pd.DataFrame(results)

# Function to calculate average UV index over different time periods - Taken directly from original notebook
def calculate_lagged_UV(UV_index, bacteria):
    """Calculate average UV index in 24h, 48h, and 72h before each bacteria sample."""
    lag_hours = [24, 48, 72]
    results = []

    for i, row in bacteria.iterrows():
        sample_time = row['Date']
        new_row = row.copy()

        for h in lag_hours:
            start = sample_time - pd.Timedelta(hours=h)
            end = sample_time
            mean_uv = UV_index[(UV_index['Date'] >= start) & (UV_index['Date'] < end)]['UVIndex'].mean()
            new_row[f'Average UV_last_{h}h'] = mean_uv

        results.append(new_row)

    return pd.DataFrame(results)

# Function to calculate average wind speed over different time periods - Taken directly from original notebook
def calculate_lagged_Wind(Wind_speed, bacteria):
    """Calculate average wind speed in 24h, 48h, and 72h before each bacteria sample."""
    lag_hours = [24, 48, 72]
    results = []

    for i, row in bacteria.iterrows():
        sample_time = row['Date']
        new_row = row.copy()

        for h in lag_hours:
            start = sample_time - pd.Timedelta(hours=h)
            end = sample_time
            mean_wind = Wind_speed[(Wind_speed['Date'] >= start) & (Wind_speed['Date'] < end)]['Wind_Sp'].mean()
            new_row[f'Average WindSpeed_last_{h}h'] = mean_wind

        results.append(new_row)

    return pd.DataFrame(results)

# Load data
with st.spinner("Loading data..."):
    Bacteria = load_data()

# Convert Date column to datetime 
if 'Date' in Bacteria.columns and not pd.api.types.is_datetime64_any_dtype(Bacteria['Date']):
    Bacteria['Date'] = pd.to_datetime(Bacteria['Date'])

# ---------- SIDEBAR UI CONTROLS ----------
# The sidebar contains filters that allow users to customize the dashboard view

st.sidebar.title("Filters")

# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=[Bacteria['Date'].min().date(), Bacteria['Date'].max().date()],
    min_value=Bacteria['Date'].min().date(),
    max_value=Bacteria['Date'].max().date()
)

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_data = Bacteria[(Bacteria['Date'].dt.date >= start_date) & 
                           (Bacteria['Date'].dt.date <= end_date)]
else:
    filtered_data = Bacteria

# Bacteria type selection
bacteria_type = st.sidebar.selectbox(
    "Select Bacteria Type",
    ["E. coli (EC)", "Intestinal Enterococci (IE)"]
)

bacteria_col = "Site 1 EC Inv" if bacteria_type == "E. coli (EC)" else "Site 1 IE Inv"

# Environmental factor selection
env_factor = st.sidebar.selectbox(
    "Select Environmental Factor",
    ["Rainfall", "Tide", "Sewage Discharge", "UV Index", "Wind Speed"]
)

lag_period = st.sidebar.selectbox(
    "Select Lag Period",
    ["24h", "48h", "72h"]
)

# Mapping of selections to column names
factor_mapping = {
    "Rainfall": f"Rainfall_last_{lag_period}",
    "Sewage Discharge": f"Average Discharge_last_{lag_period}",
    "UV Index": f"Average UV_last_{lag_period}",
    "Wind Speed": f"Average WindSpeed_last_{lag_period}",
    "Tide": "Tide Astronomical (MaOD)"
}

factor_col = factor_mapping[env_factor]

# ---------- MAIN DASHBOARD TABS ----------
# The dashboard is organized into tabs for different types of analysis

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Bacteria vs Environmental Factors", "Correlation Analysis", "Regression Analysis"])

# ---------- TAB 1: OVERVIEW ----------
# Provides a high-level view of the data with basic statistics and time series plots

with tab1:
    st.header("Data Overview")
    
    # Basic statistics shown in metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=f"Average {bacteria_type}",
            value=f"{filtered_data[bacteria_col].mean():.1f} CFU"
        )
    
    with col2:
        st.metric(
            label=f"Average {env_factor}",
            value=f"{filtered_data[factor_col].mean():.2f}"
        )
    
    with col3:
        st.metric(
            label=f"Max {bacteria_type}",
            value=f"{filtered_data[bacteria_col].max():.1f} CFU"
        )
    
    with col4:
        st.metric(
            label=f"Min {bacteria_type}",
            value=f"{filtered_data[bacteria_col].min():.1f} CFU"
        )
    
    # Time series plot of bacteria levels - Concept taken from original notebook time series plots
    st.subheader(f"{bacteria_type} Time Series")
    
    fig = px.line(
        filtered_data, 
        x='Date', 
        y=bacteria_col,
        title=f"{bacteria_type} Levels Over Time"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series comparing bacteria with environmental factor - Based on dual-axis plots from original notebook
    st.subheader(f"{bacteria_type} vs {env_factor} Time Series")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data[bacteria_col],
            name=bacteria_type,
            line=dict(color='red', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data[factor_col],
            name=env_factor,
            line=dict(color='blue', width=2, dash='dot')
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title=f"{bacteria_type} vs {env_factor} Over Time",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text=f"{bacteria_type} (CFU)", secondary_y=False)
    fig.update_yaxes(title_text=f"{env_factor}", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data preview table
    st.subheader("Data Preview")
    st.dataframe(filtered_data.sort_values('Date'), use_container_width=True)

# ---------- TAB 2: BACTERIA VS ENVIRONMENTAL FACTORS ----------
# Analyzes the relationship between bacteria levels and selected environmental factors

with tab2:
    st.header("Bacteria vs Environmental Factors")
    
    # Scatter plot with regression line - Based on regression plots from original notebook
    st.subheader(f"{bacteria_type} vs {env_factor}")
    
    fig = px.scatter(
        filtered_data, 
        x=factor_col, 
        y=bacteria_col,
        trendline="ols",
        labels={
            factor_col: f"{env_factor}",
            bacteria_col: f"{bacteria_type} (CFU)"
        },
        title=f"Relationship between {bacteria_type} and {env_factor}"
    )
    
    # Calculate correlation and add it to the plot - Pearson correlations from original notebook
    corr, p_val = pearsonr(filtered_data[factor_col].fillna(0), filtered_data[bacteria_col].fillna(0))
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.05, y=0.95,
        text=f"Correlation: {corr:.2f}<br>p-value: {p_val:.4f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Compare the effect of different lag periods - Based on lag period analysis in original notebook
    st.subheader("Impact of Time Lag")
    
    # Calculate correlation for different lag periods
    lag_data = []
    
    for lag in ["24h", "48h", "72h"]:
        if env_factor != "Tide":  # Skip for Tide as it doesn't have lag periods
            if env_factor == "Rainfall":
                col = f"Rainfall_last_{lag}"
            elif env_factor == "Sewage Discharge":
                col = f"Average Discharge_last_{lag}"
            elif env_factor == "UV Index":
                col = f"Average UV_last_{lag}"
            elif env_factor == "Wind Speed":
                col = f"Average WindSpeed_last_{lag}"
            
            if col in filtered_data.columns:
                corr, p_val = pearsonr(filtered_data[bacteria_col].fillna(0), 
                                      filtered_data[col].fillna(0))
                lag_data.append({
                    "Lag Period": lag,
                    "Correlation Coefficient": corr,
                    "p-value": p_val,
                    "Significance": "Significant" if p_val < 0.05 else "Not Significant"
                })
    
    if lag_data:
        lag_df = pd.DataFrame(lag_data)
        
        # Create a bar chart using go.Figure instead of px.bar to avoid DataFrame issues
        fig = go.Figure()
        
        for idx, row in lag_df.iterrows():
            color = "blue" if row["Significance"] == "Significant" else "gray"
            fig.add_trace(go.Bar(
                x=[row["Lag Period"]], 
                y=[row["Correlation Coefficient"]],
                name=row["Lag Period"],
                marker_color=color
            ))
        
        fig.update_layout(
            title=f"Impact of Time Lag on Correlation with {bacteria_type}",
            xaxis_title="Lag Period",
            yaxis_title="Pearson Correlation (r)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No lag analysis available for this environmental factor.")
    
    # Compare all environmental factors - Based on the correlation comparison in the original notebook
    st.subheader("Comparison of All Environmental Factors")
    
    # Calculate correlation for each environmental factor
    env_factors = ["Rainfall", "Sewage Discharge", "UV Index", "Wind Speed", "Tide"]
    factor_data = []
    
    for factor in env_factors:
        if factor == "Tide":
            col = "Tide Astronomical (MaOD)"
        else:
            if factor == "Rainfall":
                col = f"Rainfall_last_{lag_period}"
            elif factor == "Sewage Discharge":
                col = f"Average Discharge_last_{lag_period}"
            elif factor == "UV Index":
                col = f"Average UV_last_{lag_period}"
            elif factor == "Wind Speed":
                col = f"Average WindSpeed_last_{lag_period}"
        
        if col in filtered_data.columns:
            corr, p_val = pearsonr(filtered_data[bacteria_col].fillna(0), 
                                  filtered_data[col].fillna(0))
            factor_data.append({
                "Environmental Factor": factor,
                "Correlation Coefficient": corr,
                "p-value": p_val,
                "Significance": "Significant" if p_val < 0.05 else "Not Significant"
            })
    
    factor_df = pd.DataFrame(factor_data)
    
    # Create bar chart of correlations for all factors
    fig = go.Figure()
    
    for idx, row in factor_df.iterrows():
        color = "blue" if row["Significance"] == "Significant" else "gray"
        fig.add_trace(go.Bar(
            x=[row["Environmental Factor"]], 
            y=[row["Correlation Coefficient"]],
            name=row["Environmental Factor"],
            marker_color=color
        ))
    
    # Add reference line at y=0
    fig.add_shape(
        type="line",
        x0=-0.5, 
        y0=0, 
        x1=len(env_factors) - 0.5, 
        y1=0,
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig.update_layout(
        title=f"Correlation of Environmental Factors with {bacteria_type}",
        xaxis_title="Environmental Factor",
        yaxis_title="Pearson Correlation (r)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 3: CORRELATION ANALYSIS ----------
# Provides detailed correlation analysis between multiple variables

with tab3:
    st.header("Correlation Analysis")
    
    # User-selectable factors for correlation matrix
    factors_to_include = st.multiselect(
        "Select Factors to Include in Correlation Matrix",
        options=["Bacteria EC", "Bacteria IE", "Rainfall", "Tide", "Sewage Discharge", "UV Index", "Wind Speed"],
        default=["Bacteria EC", "Bacteria IE", "Rainfall", "Sewage Discharge"]
    )
    
    # Map friendly names to actual column names
    columns_mapping = {
        "Bacteria EC": "Site 1 EC Inv",
        "Bacteria IE": "Site 1 IE Inv",
        "Rainfall": f"Rainfall_last_{lag_period}",
        "Tide": "Tide Astronomical (MaOD)",
        "Sewage Discharge": f"Average Discharge_last_{lag_period}",
        "UV Index": f"Average UV_last_{lag_period}",
        "Wind Speed": f"Average WindSpeed_last_{lag_period}"
    }
    
    selected_columns = [columns_mapping[factor] for factor in factors_to_include if columns_mapping[factor] in filtered_data.columns]
    
    if selected_columns:
        # Create correlation matrix - Based on correlation heatmaps in original notebook
        correlation_df = filtered_data[selected_columns].corr()
        
        # Create heatmap visualization
        fig = px.imshow(
            correlation_df,
            x=correlation_df.columns,
            y=correlation_df.columns,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=True,
            title="Correlation Matrix"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed correlation statistics - Based on Pearson correlation tests in original notebook
        st.subheader("Detailed Correlation Statistics")
        
        # Calculate pairwise correlations with significance tests
        detailed_corr = []
        
        for i, col1 in enumerate(selected_columns):
            for col2 in selected_columns[i+1:]:
                corr, p_val = pearsonr(filtered_data[col1].fillna(0), filtered_data[col2].fillna(0))
                
                # Get display names
                col1_name = next(k for k, v in columns_mapping.items() if v == col1)
                col2_name = next(k for k, v in columns_mapping.items() if v == col2)
                
                detailed_corr.append({
                    "Factor 1": col1_name,
                    "Factor 2": col2_name,
                    "Correlation Coefficient": corr,
                    "p-value": p_val,
                    "Significance": "Significant" if p_val < 0.05 else "Not Significant"
                })
        
        detailed_corr_df = pd.DataFrame(detailed_corr)
        detailed_corr_df = detailed_corr_df.sort_values(by="Correlation Coefficient", key=abs, ascending=False)
        
        st.dataframe(detailed_corr_df, use_container_width=True)
    else:
        st.warning("Please select at least one factor to include in the correlation matrix.")

# ---------- TAB 4: REGRESSION ANALYSIS ----------
# Provides simple regression analysis without prediction functionality

with tab4:
    st.header("Regression Analysis")
    
    # Multiple regression analysis - Based on regression models in original notebook
    st.subheader("Multiple Regression Model")
    
    # User-selectable predictors for regression
    predictors = st.multiselect(
        "Select Predictors for Multiple Regression",
        options=["Rainfall", "Tide", "Sewage Discharge", "UV Index", "Wind Speed"],
        default=["Rainfall", "Sewage Discharge", "UV Index"]
    )
    
    predictor_columns = [columns_mapping[factor] for factor in predictors if columns_mapping[factor] in filtered_data.columns]
    
    if predictor_columns:
        # Prepare data for regression - Similar to original notebook's OLS regression code
        X = filtered_data[predictor_columns].copy()
        y = filtered_data[bacteria_col].copy()
        
        # Drop rows with missing values
        valid_data = pd.concat([X, y], axis=1).dropna()
        
        if len(valid_data) > len(predictor_columns) + 1:  # Check if enough data points
            X_clean = valid_data[predictor_columns]
            y_clean = valid_data[bacteria_col]
            
            # Add constant term
            X_with_const = sm.add_constant(X_clean)
            
            # Fit OLS model - Direct implementation from original notebook
            model = sm.OLS(y_clean, X_with_const).fit()
            
            # Display results
            st.text(model.summary().as_text())
            
            # Extract key metrics
            r_squared = model.rsquared
            adj_r_squared = model.rsquared_adj
            f_stat = model.fvalue
            f_pvalue = model.f_pvalue
            
            # Display key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("R-squared", f"{r_squared:.3f}")
            
            with col2:
                st.metric("Adjusted R-squared", f"{adj_r_squared:.3f}")
            
            with col3:
                st.metric("F-statistic", f"{f_stat:.2f}")
            
            with col4:
                st.metric("F-test p-value", f"{f_pvalue:.4f}")
            
            # Feature importance visualization - Extension of regression analysis from original notebook
            st.subheader("Feature Importance")
            
            # Standardize predictors to get comparable coefficients
            X_std = (X_clean - X_clean.mean()) / X_clean.std()
            X_std = sm.add_constant(X_std)
            
            # Fit standardized model
            model_std = sm.OLS(y_clean, X_std).fit()
            
            # Get standardized coefficients (excluding constant)
            std_coeffs = model_std.params[1:].abs()
            std_coeffs = std_coeffs / std_coeffs.sum()  # Normalize to sum to 1
            
            # Create feature importance bar chart
            importance_df = pd.DataFrame({
                'Feature': [factor for factor in predictors if columns_mapping[factor] in X_clean.columns],
                'Importance': std_coeffs.values
            })
            
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Use go.Figure instead of px.bar to avoid issues
            fig = go.Figure()
            
            for idx, row in importance_df.iterrows():
                fig.add_trace(go.Bar(
                    x=[row["Feature"]], 
                    y=[row["Importance"]],
                    name=row["Feature"],
                    marker_color="blue"
                ))
            
            fig.update_layout(
                title="Relative Feature Importance",
                xaxis_title="Feature",
                yaxis_title="Importance"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data points for regression analysis. Please select a wider date range or different variables.")
    else:
        st.warning("Please select at least one predictor for the regression model.")

# Footer
st.markdown("---")
st.write("Data source: Environmental Agency - Water Quality Monitoring System (2023)")

# Run the app
# To run: streamlit run app.py