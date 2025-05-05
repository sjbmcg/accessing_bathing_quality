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
from scipy import stats

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

@st.cache_data
def load_data():
    """
    Load and process the water quality data.
    Uses caching to avoid reprocessing on every interaction.
    """
    try:
        # Try to load processed data first
        return pd.read_csv("Bacteria.csv", parse_dates=["Date"])
    except FileNotFoundError:
        # If not found, process the raw data
        xls = pd.ExcelFile('Anonymised data 2023 mapped.xlsx')
        
        # Create dataset for each excel sheet - Following original notebook
        Principle_Data = pd.read_excel(xls, 0)
        Investigation_samples = pd.read_excel(xls, 1)
        Tide_data = pd.read_excel(xls, 3)
        STW_flow_UV = pd.read_excel(xls, 4)
        UV_Wind = pd.read_excel(xls, 7)
        
        # Create datetime columns
        Principle_Data['datetime'] = pd.to_datetime(Principle_Data['Date'].dt.strftime('%Y-%m-%d') + ' ' + Principle_Data['Time GMT'])
        Investigation_samples['datetime'] = pd.to_datetime(Investigation_samples['Date'].dt.strftime('%Y-%m-%d') + ' ' + Investigation_samples['Time GMT'])
        
        # Process data - Column renaming from original notebook
        Bacteria_with_rain = calculate_lagged_rainfall(Principle_Data, Investigation_samples)
        Bacteria_with_rain.rename(columns={'Date': 'Date_NA'}, inplace=True)
        Bacteria_with_rain.rename(columns={'datetime': 'Date'}, inplace=True)
        
        # Add tide data
        Tide_data2 = Tide_data[['Date', 'Tide Astronomical (MaOD)']]
        Bacteria = Bacteria_with_rain.merge(Tide_data2, on='Date', how='left')
        
        # Add sewage discharge data
        Bacteria = calculate_lagged_sewage(STW_flow_UV, Bacteria)
        
        # Add UV index data
        Bacteria = calculate_lagged_UV(UV_Wind, Bacteria)
        
        # Add wind speed data
        Bacteria = calculate_lagged_Wind(UV_Wind, Bacteria)
        
        # Save processed data
        Bacteria.to_csv("Bacteria.csv", index=False)
        
        return Bacteria

# Function to calculate cumulative rainfall over different time periods
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

# Function to calculate average sewage discharge over different time periods
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

# Function to calculate average UV index over different time periods
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

# Function to calculate average wind speed over different time periods
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

# ---------- SIDEBAR UI CONTROLS ----------

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

# Environment factor colors - consistent color mapping
env_colors = {
    "Rainfall": "#0072B2",         # Blue
    "Tide": "#009E73",             # Green
    "Sewage Discharge": "#D55E00", # Orange/brown
    "UV Index": "#E69F00",         # Yellow/amber
    "Wind Speed": "#56B4E9",       # Light blue
    "E. coli (EC)": "#CC79A7",     # Pink
    "Intestinal Enterococci (IE)": "#CC79A7"  # Pink (same for both bacteria types)
}

# ---------- MAIN DASHBOARD TABS ----------

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Bacteria vs Environmental Factors", "Correlation Analysis", "Regression Analysis", "Statistical Tests"])

# ---------- TAB 1: OVERVIEW ----------

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
    
    # Time series plot of bacteria levels
    st.subheader(f"{bacteria_type} Time Series")
    
    fig = px.line(
        filtered_data, 
        x='Date', 
        y=bacteria_col,
        title=f"{bacteria_type} Levels Over Time",
        color_discrete_sequence=[env_colors[bacteria_type]]
    )
    
    # Improve legend formatting
    fig.update_layout(
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis_title="CFU (Colony Forming Units)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series comparing bacteria with environmental factor
    st.subheader(f"{bacteria_type} vs {env_factor} Time Series")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bacteria trace with consistent color
    fig.add_trace(
        go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data[bacteria_col],
            name=f"{bacteria_type}",
            line=dict(
                color=env_colors[bacteria_type], 
                width=2
            )
        )
    )
    
    # Add environmental factor trace with consistent color
    fig.add_trace(
        go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data[factor_col],
            name=f"{env_factor}",
            line=dict(
                color=env_colors[env_factor], 
                width=2, 
                dash='dot'
            )
        ),
        secondary_y=True
    )
    
    # Improve layout and legend
    fig.update_layout(
        title=f"{bacteria_type} vs {env_factor} Over Time",
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_yaxes(title_text=f"{bacteria_type} (CFU)", secondary_y=False)
    fig.update_yaxes(title_text=f"{env_factor}", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Overall correlations with bacteria heatmap
    st.subheader("Overall Correlation Heatmap")
    
    # Create correlation matrix of all environmental factors
    correlation_columns = [
        'Site 1 EC Inv', 'Site 1 IE Inv',
        'Rainfall_last_24h', 'Rainfall_last_48h', 'Rainfall_last_72h',
        'Tide Astronomical (MaOD)',
        'Average Discharge_last_24h', 'Average Discharge_last_48h', 'Average Discharge_last_72h',
        'Average UV_last_24h', 'Average UV_last_48h', 'Average UV_last_72h',
        'Average WindSpeed_last_24h', 'Average WindSpeed_last_48h', 'Average WindSpeed_last_72h'
    ]
    
    # Only include columns that exist in our filtered data
    corr_columns = [col for col in correlation_columns if col in filtered_data.columns]
    
    # Create correlation matrix
    corr_matrix = filtered_data[corr_columns].corr()
    
    # Create a heatmap using Plotly
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title="Correlation Matrix of All Factors"
    )
    
    fig.update_layout(
        width=900,
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data preview table
    st.subheader("Data Preview")
    st.dataframe(filtered_data.sort_values('Date'), use_container_width=True)

# ---------- TAB 2: BACTERIA VS ENVIRONMENTAL FACTORS ----------

with tab2:
    st.header("Bacteria vs Environmental Factors")
    
    # Scatter plot with regression line
    st.subheader(f"{bacteria_type} vs {env_factor}")
    
    # Create scatter plot with consistent colors
    fig = px.scatter(
        filtered_data, 
        x=factor_col, 
        y=bacteria_col,
        trendline="ols",
        labels={
            factor_col: f"{env_factor}",
            bacteria_col: f"{bacteria_type} (CFU)"
        },
        title=f"Relationship between {bacteria_type} and {env_factor}",
        color_discrete_sequence=[env_colors[bacteria_type]]
    )
    
    # Change trendline color to a darker shade for better visibility
    for trace in fig.data:
        if trace.mode == "lines":
            trace.line.color = "#882255"  # Darker version of bacteria color
    
    # Calculate correlation and add it to the plot
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
    
    # Update legend
    fig.update_layout(
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Compare the effect of different lag periods
    st.subheader("Impact of Time Lag")
    
    if env_factor != "Tide":  # Skip for Tide as it doesn't have lag periods
        # Calculate correlation for different lag periods
        lag_data = []
        
        for lag in ["24h", "48h", "72h"]:
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
            
            # Create a bar chart with consistent colors
            fig = go.Figure()
            
            for idx, row in lag_df.iterrows():
                # Use different color intensities based on significance
                color = env_colors[env_factor] if row["Significance"] == "Significant" else "#CCCCCC"
                fig.add_trace(go.Bar(
                    x=[row["Lag Period"]], 
                    y=[row["Correlation Coefficient"]],
                    name=row["Lag Period"],
                    marker_color=color,
                    text=[f"{row['Correlation Coefficient']:.2f}<br>{row['Significance']}"],
                    textposition="auto"
                ))
            
            # Improve layout
            fig.update_layout(
                title=f"Impact of Time Lag on Correlation with {bacteria_type}",
                xaxis_title="Lag Period",
                yaxis_title="Pearson Correlation (r)",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No lag analysis available for this environmental factor.")
    
    # Compare all environmental factors
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
    
    # Create bar chart with consistent colors for each factor
    fig = go.Figure()
    
    for idx, row in factor_df.iterrows():
        factor = row["Environmental Factor"]
        # Use colors from our environmental factor mapping
        color = env_colors[factor] if row["Significance"] == "Significant" else "#CCCCCC"
        
        fig.add_trace(go.Bar(
            x=[factor], 
            y=[row["Correlation Coefficient"]],
            name=factor,
            marker_color=color,
            text=[f"{row['Correlation Coefficient']:.2f}<br>{row['Significance']}"],
            textposition="auto"
        ))
    
    # Add reference line at y=0
    fig.add_shape(
        type="line",
        x0=-0.5, 
        y0=0, 
        x1=len(env_factors) - 0.5, 
        y1=0,
        line=dict(color="black", width=1, dash="dash")
    )
    
    # Improve layout
    fig.update_layout(
        title=f"Correlation of Environmental Factors with {bacteria_type}",
        xaxis_title="Environmental Factor",
        yaxis_title="Pearson Correlation (r)",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add scatter plots with 95% confidence intervals
    st.subheader("Scatter Plots with Confidence Intervals")
    
    # Create two columns for side-by-side plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Create scatter plot with regression line and confidence interval
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.regplot(
            data=filtered_data,
            x=factor_col,
            y=bacteria_col,
            scatter_kws={'alpha': 0.6},
            line_kws={'color': 'red'},
            ci=95
        )
        
        plt.title(f"{bacteria_type} vs {env_factor} with 95% CI")
        plt.xlabel(f"{env_factor}")
        plt.ylabel(f"{bacteria_type} (CFU)")
        plt.tight_layout()
        
        st.pyplot(fig)
    
    with col2:
        # If we're looking at Rainfall factor, show alternative visualization
        alt_factor = "Rainfall" if env_factor != "Rainfall" else "Sewage Discharge"
        alt_col = f"{alt_factor}_last_{lag_period}" if alt_factor != "Tide" else "Tide Astronomical (MaOD)"
        
        if alt_col in filtered_data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sns.regplot(
                data=filtered_data,
                x=alt_col,
                y=bacteria_col,
                scatter_kws={'alpha': 0.6},
                line_kws={'color': 'red'},
                ci=95
            )
            
            plt.title(f"{bacteria_type} vs {alt_factor} with 95% CI")
            plt.xlabel(f"{alt_factor}")
            plt.ylabel(f"{bacteria_type} (CFU)")
            plt.tight_layout()
            
            st.pyplot(fig)

# ---------- TAB 3: CORRELATION ANALYSIS ----------

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
        # Create correlation matrix
        correlation_df = filtered_data[selected_columns].corr()
        
        # Rename columns for better display
        display_names = {}
        for factor, col in columns_mapping.items():
            if col in correlation_df.columns:
                display_names[col] = factor
        
        correlation_df_display = correlation_df.rename(columns=display_names, index=display_names)
        
        # Create improved heatmap visualization with better color scheme
        fig = px.imshow(
            correlation_df_display,
            x=correlation_df_display.columns,
            y=correlation_df_display.columns,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=True,
            title="Correlation Matrix"
        )
        
        # Improve layout
        fig.update_layout(
            xaxis_title="",
            yaxis_title="",
            coloraxis_colorbar=dict(
                title="Correlation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed correlation statistics
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
        
        # Apply conditional formatting to the displayed dataframe
        st.dataframe(
            detailed_corr_df.style.format({
                "Correlation Coefficient": "{:.3f}",
                "p-value": "{:.4f}"
            }),
            use_container_width=True
        )
        
        # Add lag periods analysis
        st.subheader("Lag Period Analysis for Bacteria")
        
        # For each environmental factor with lag periods, create a heatmap
        env_factors_with_lag = ["Rainfall", "Sewage Discharge", "UV Index", "Wind Speed"]
        bacteria_types = ["E. coli", "IE"]
        
        # Create a DataFrame to store correlation results
        lag_correlation_data = []
        
        for bacteria in bacteria_types:
            bacteria_col = "Site 1 EC Inv" if bacteria == "E. coli" else "Site 1 IE Inv"
            
            for factor in env_factors_with_lag:
                for lag in ["24h", "48h", "72h"]:
                    if factor == "Rainfall":
                        col = f"Rainfall_last_{lag}"
                    elif factor == "Sewage Discharge":
                        col = f"Average Discharge_last_{lag}"
                    elif factor == "UV Index":
                        col = f"Average UV_last_{lag}"
                    elif factor == "Wind Speed":
                        col = f"Average WindSpeed_last_{lag}"
                    
                    if col in filtered_data.columns:
                        corr, p_val = pearsonr(filtered_data[bacteria_col].fillna(0), 
                                              filtered_data[col].fillna(0))
                        
                        lag_correlation_data.append({
                            "Bacteria Type": bacteria,
                            "Environmental Factor": factor,
                            "Lag Period": lag,
                            "Correlation": corr,
                            "p-value": p_val,
                            "Significance": "Significant" if p_val < 0.05 else "Not Significant"
                        })
        
        lag_corr_df = pd.DataFrame(lag_correlation_data)
        
        # Create a heatmap for lag period analysis
        st.subheader("Correlation Heatmap by Lag Period")
        
        bacteria_choice = st.radio("Select Bacteria Type for Heatmap", ["E. coli", "IE"])
        
        # Filter data for the selected bacteria type
        filtered_lag_corr = lag_corr_df[lag_corr_df["Bacteria Type"] == bacteria_choice]
        
        # Create a pivot table for the heatmap
        heatmap_data = filtered_lag_corr.pivot(
            index="Environmental Factor", 
            columns="Lag Period", 
            values="Correlation"
        )
        
        # Create heatmap
        fig = px.imshow(
            heatmap_data,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title=f"Correlation of {bacteria_choice} with Environmental Factors by Lag Period"
        )
        
        fig.update_layout(
            xaxis_title="Lag Period",
            yaxis_title="Environmental Factor",
            coloraxis_colorbar=dict(
                title="Correlation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one factor to include in the correlation matrix.")

# ---------- TAB 4: REGRESSION ANALYSIS ----------
# Based on the regression models from the original notebook

with tab4:
    st.header("Regression Analysis")
    
    # Multiple regression analysis
    st.subheader("Multiple Regression Model")
    
    # User-selectable predictors for regression
    predictors = st.multiselect(
        "Select Predictors for Multiple Regression",
        options=["Rainfall", "Tide", "Sewage Discharge", "UV Index", "Wind Speed"],
        default=["Rainfall", "Sewage Discharge", "UV Index"]
    )
    
    predictor_columns = [columns_mapping[factor] for factor in predictors if columns_mapping[factor] in filtered_data.columns]
    
    if predictor_columns:
        # Prepare data for regression
        X = filtered_data[predictor_columns].copy()
        y = filtered_data[bacteria_col].copy()
        
        # Drop rows with missing values
        valid_data = pd.concat([X, y], axis=1).dropna()
        
        if len(valid_data) > len(predictor_columns) + 1:  # Check if enough data points
            X_clean = valid_data[predictor_columns]
            y_clean = valid_data[bacteria_col]
            
            # Add constant term
            X_with_const = sm.add_constant(X_clean)
            
            # Fit OLS model
            model = sm.OLS(y_clean, X_with_const).fit()
            
            # Display results
            st.text(model.summary().as_text())
            
            # Extract key metrics
            r_squared = model.rsquared
            adj_r_squared = model.rsquared_adj
            f_stat = model.fvalue
            f_pvalue = model.f_pvalue
            
            # Display key metrics in columns with improved styling
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="R-squared",
                    value=f"{r_squared:.3f}",
                    delta=None,
                    delta_color="normal"
                )
            
            with col2:
                st.metric(
                    label="Adjusted R-squared",
                    value=f"{adj_r_squared:.3f}",
                    delta=None,
                    delta_color="normal"
                )
            
            with col3:
                st.metric(
                    label="F-statistic",
                    value=f"{f_stat:.2f}",
                    delta=None,
                    delta_color="normal"
                )
            
            with col4:
                st.metric(
                    label="F-test p-value",
                    value=f"{f_pvalue:.4f}",
                    delta=None,
                    delta_color="normal"
                )
            
            # Feature importance visualization
            st.subheader("Feature Importance")
            
            # Standardize predictors to get comparable coefficients
            X_std = (X_clean - X_clean.mean()) / X_clean.std()
            X_std = sm.add_constant(X_std)
            
            # Fit standardized model
            model_std = sm.OLS(y_clean, X_std).fit()
            
            # Get standardized coefficients (excluding constant)
            std_coeffs = model_std.params[1:].abs()
            std_coeffs = std_coeffs / std_coeffs.sum()  # Normalize to sum to 1
            
            # Create feature importance bar chart with consistent colors
            importance_df = pd.DataFrame({
                'Feature': [factor for factor in predictors if columns_mapping[factor] in X_clean.columns],
                'Importance': std_coeffs.values
            })
            
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Create bar chart with consistent colors for each feature
            fig = go.Figure()
            
            for idx, row in importance_df.iterrows():
                factor = row["Feature"]
                fig.add_trace(go.Bar(
                    x=[factor], 
                    y=[row["Importance"]],
                    name=factor,
                    marker_color=env_colors[factor],
                    text=[f"{row['Importance']:.3f}"],
                    textposition="auto"
                ))
            
            # Improve layout
            fig.update_layout(
                title="Relative Feature Importance",
                xaxis_title="Feature",
                yaxis_title="Importance (Normalized)",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # UV Index specific analysis (from the notebook)
            if "UV Index" in predictors:
                st.subheader("UV Index Impact Analysis")
                
                uv_cols = [col for col in predictor_columns if "UV" in col]
                if uv_cols:
                    # Prepare data for UV regression
                    X_uv = filtered_data[uv_cols].copy()
                    y_uv = filtered_data[bacteria_col].copy()
                    
                    # Drop rows with missing values
                    valid_data_uv = pd.concat([X_uv, y_uv], axis=1).dropna()
                    
                    if len(valid_data_uv) > len(uv_cols) + 1:
                        X_uv_clean = valid_data_uv[uv_cols]
                        y_uv_clean = valid_data_uv[bacteria_col]
                        
                        # Add constant term
                        X_uv_with_const = sm.add_constant(X_uv_clean)
                        
                        # Fit OLS model
                        uv_model = sm.OLS(y_uv_clean, X_uv_with_const).fit()
                        
                        # Display results
                        st.text("UV Index Regression Model Results:")
                        st.text(uv_model.summary().as_text())
                        
                        # Create a visualization of UV impact
                        st.subheader("UV Index Effect on Bacteria")
                        
                        # Create scatter plot with regression line
                        fig = px.scatter(
                            valid_data_uv, 
                            x=uv_cols[0] if len(uv_cols) > 0 else None,
                            y=bacteria_col,
                            trendline="ols",
                            title=f"Impact of UV Index on {bacteria_type}",
                            color_discrete_sequence=["#E69F00"]  # UV color
                        )
                        
                        fig.update_layout(
                            xaxis_title="UV Index",
                            yaxis_title=f"{bacteria_type} (CFU)"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough data points for UV Index regression analysis.")
        else:
            st.warning("Not enough data points for regression analysis. Please select a wider date range or different variables.")
    else:
        st.warning("Please select at least one predictor for the regression model.")