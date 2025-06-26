"""
Machine Learning Stock Prediction Dashboard
Built with Streamlit for interactive visualization and model management
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to Python path
sys.path.append('./src')

# Import your custom modules
try:
    import machine_learning_data as mld
    import machine_learning_plotting as mlp
    import machine_learning_training as mlt
    import machine_learning_dataloading as mldl
except ImportError as e:
    st.error(f"Error importing custom modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="ML Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🚀 Machine Learning Stock Prediction Dashboard")
st.markdown("""
This dashboard provides comprehensive analysis and prediction capabilities for financial markets using LSTM neural networks and sentiment analysis.
""")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Stock selection
available_stocks = {
    "S&P 500": "^GSPC",
    "Apple": "AAPL", 
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "NVIDIA": "NVDA"
}

selected_stock_name = st.sidebar.selectbox(
    "Select Stock/Index:",
    options=list(available_stocks.keys()),
    index=0
)
selected_stock = available_stocks[selected_stock_name]

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
with col2:
    end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))

# Data refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Main dashboard tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data Overview", 
    "🔍 Analysis", 
    "📈 Correlations",
    "🤖 Predictions", 
    "📰 Sentiment",
    "⚙️ Model Config"
])

# Load data function with caching
@st.cache_data
def load_financial_data():
    """Load and cache financial data"""
    try:
        # Load your existing data
        data_files = [
            './data/financial_data.csv',
            './data/gdp_data.csv', 
            './data/interest_rate_data.csv',
            './data/inflation_data.csv',
            './data/unemployment_rate_data.csv'
        ]
        
        # Check if files exist
        existing_files = [f for f in data_files if os.path.exists(f)]
        
        if not existing_files:
            st.error("Data files not found. Please ensure data is downloaded.")
            return None, None
            
        tensor_data, loaded_data = mldl.create_tensor_from_csvs(existing_files)
        
        # Clean up loaded_data for Streamlit compatibility
        cleaned_data = {}
        for key, df in loaded_data.items():
            if isinstance(df, pd.DataFrame):
                # Make a copy to avoid modifying original data
                clean_df = df.copy()
                
                # Handle date columns for Arrow compatibility
                for col in clean_df.columns:
                    if 'date' in col.lower():
                        try:
                            # Convert to datetime then to string format
                            clean_df[col] = pd.to_datetime(clean_df[col]).dt.strftime('%Y-%m-%d')
                        except:
                            # If conversion fails, keep as string
                            clean_df[col] = clean_df[col].astype(str)
                    elif clean_df[col].dtype == 'object':
                        # Convert other object columns to string
                        clean_df[col] = clean_df[col].astype(str)
                
                cleaned_data[key] = clean_df
            else:
                cleaned_data[key] = df
                
        return tensor_data, cleaned_data
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Load sentiment data
@st.cache_data
def load_sentiment_data(stock_symbol):
    """Load sentiment data for specific stock"""
    try:
        sentiment_file = f'./data/{stock_symbol.lower()}_sentiment_gpu_parallel_filtered.csv'
        if stock_symbol == '^GSPC':
            sentiment_file = './data/SP500_sentiment_gpu_parallel_filtered.csv'
        
        if os.path.exists(sentiment_file):
            return pd.read_csv(sentiment_file)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading sentiment data: {e}")
        return None

# Debug information (can be hidden in production)
with st.sidebar.expander("🔧 Debug Info"):
    if st.button("Show Data Structure"):
        tensor_data, loaded_data = load_financial_data()
        if loaded_data:
            st.write("**Loaded datasets:**")
            for key, df in loaded_data.items():
                st.write(f"- {key}: {df.shape} - {list(df.columns)}")
        else:
            st.write("No data loaded")

# Tab 1: Data Overview
with tab1:
    st.header("📊 Financial Data Overview")
    
    # Load data
    tensor_data, loaded_data = load_financial_data()
    
    if loaded_data is not None:
        # Display data summary
        col1, col2, col3, col4 = st.columns(4)
        
        if 'financial_data.csv' in loaded_data:
            financial_df = loaded_data['financial_data.csv']
            
            with col1:
                st.metric("Current Price", f"${financial_df['Close'].iloc[-1]:.2f}")
            with col2:
                price_change = financial_df['Close'].iloc[-1] - financial_df['Close'].iloc[-2]
                st.metric("Daily Change", f"${price_change:.2f}", delta=f"{(price_change/financial_df['Close'].iloc[-2]*100):.2f}%")
            with col3:
                st.metric("Volume", f"{financial_df['Volume'].iloc[-1]:,.0f}")
            with col4:
                volatility = financial_df['Close'].pct_change().std() * np.sqrt(252) * 100
                st.metric("Volatility (252d)", f"{volatility:.2f}%")
        
        # Financial data plot
        st.subheader("Price History")
        if 'financial_data.csv' in loaded_data:
            try:
                fig = go.Figure()
                df = loaded_data['financial_data.csv']
                
                # Handle date column properly
                date_col = None
                for col in df.columns:
                    if 'date' in col.lower():
                        date_col = col
                        break
                
                if date_col is not None:
                    # Convert date column back to datetime for plotting
                    dates = pd.to_datetime(df[date_col])
                    
                    # Candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=dates,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name=selected_stock_name
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_stock_name} Price Chart",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Date column not found in financial data")
                    
            except Exception as e:
                st.error(f"Error creating price chart: {e}")
        
        # Economic indicators
        st.subheader("Economic Indicators")
        econ_cols = st.columns(2)
        
        with econ_cols[0]:
            if 'gdp_data.csv' in loaded_data:
                try:
                    gdp_df = loaded_data['gdp_data.csv']
                    # Find date column
                    date_col = None
                    for col in gdp_df.columns:
                        if 'date' in col.lower():
                            date_col = col
                            break
                    
                    if date_col is not None:
                        gdp_dates = pd.to_datetime(gdp_df[date_col])
                        # Find GDP column
                        gdp_col = None
                        for col in gdp_df.columns:
                            if 'gdp' in col.lower():
                                gdp_col = col
                                break
                        
                        if gdp_col is not None:
                            fig_gdp = px.line(x=gdp_dates, y=gdp_df[gdp_col], title="GDP Trends")
                            fig_gdp.update_xaxes(title="Date")
                            fig_gdp.update_yaxes(title="GDP")
                            st.plotly_chart(fig_gdp, use_container_width=True)
                        else:
                            st.warning("GDP column not found")
                    else:
                        st.warning("Date column not found in GDP data")
                except Exception as e:
                    st.error(f"Error plotting GDP data: {e}")
        
        with econ_cols[1]:
            if 'inflation_data.csv' in loaded_data:
                try:
                    inflation_df = loaded_data['inflation_data.csv']
                    # Find date column
                    date_col = None
                    for col in inflation_df.columns:
                        if 'date' in col.lower():
                            date_col = col
                            break
                    
                    if date_col is not None:
                        inflation_dates = pd.to_datetime(inflation_df[date_col])
                        # Find inflation column
                        inflation_col = None
                        for col in inflation_df.columns:
                            if 'inflation' in col.lower():
                                inflation_col = col
                                break
                        
                        if inflation_col is not None:
                            fig_inflation = px.line(x=inflation_dates, y=inflation_df[inflation_col], title="Inflation Rate")
                            fig_inflation.update_xaxes(title="Date")
                            fig_inflation.update_yaxes(title="Inflation Rate")
                            st.plotly_chart(fig_inflation, use_container_width=True)
                        else:
                            st.warning("Inflation column not found")
                    else:
                        st.warning("Date column not found in inflation data")
                except Exception as e:
                    st.error(f"Error plotting inflation data: {e}")
    else:
        st.warning("Please load data first or check data files.")

# Tab 2: Analysis
with tab2:
    st.header("🔍 Exploratory Data Analysis")
    
    if loaded_data is not None:
        # Data selection
        available_datasets = list(loaded_data.keys())
        selected_dataset = st.selectbox("Select Dataset:", available_datasets)
        
        if selected_dataset:
            df = loaded_data[selected_dataset]
            
            # Ensure df is a DataFrame and handle date columns properly
            if isinstance(df, pd.Series):
                df = df.to_frame()
            
            # Convert date columns to string to avoid Arrow serialization issues
            for col in df.columns:
                if 'date' in col.lower() or df[col].dtype == 'object':
                    try:
                        # Try to convert to datetime first, then to string
                        df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        # If conversion fails, convert to string
                        df[col] = df[col].astype(str)
            
            # Statistical summary
            st.subheader("Statistical Summary")
            try:
                st.dataframe(df.describe())
            except Exception as e:
                st.error(f"Error displaying data summary: {e}")
                st.write("Data shape:", df.shape)
                st.write("Data types:", df.dtypes)
            
            # Distribution plots
            st.subheader("Data Distributions")
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) > 0:
                selected_column = st.selectbox("Select Column for Distribution:", numeric_columns)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_hist = px.histogram(df, x=selected_column, title=f"Distribution of {selected_column}")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    fig_box = px.box(df, y=selected_column, title=f"Box Plot of {selected_column}")
                    st.plotly_chart(fig_box, use_container_width=True)

# Tab 3: Correlations
with tab3:
    st.header("📈 Correlation Analysis")
    
    if loaded_data is not None:
        try:
            # Generate correlation matrix
            corr_matrix, found_vars, corr_df = mlp.create_half_correlation_plot3(loaded_data, plot=False, save=False)
            
            # Interactive correlation heatmap
            fig_corr = px.imshow(
                corr_matrix, 
                labels=dict(x="Variables", y="Variables", color="Correlation"),
                x=found_vars,
                y=found_vars,
                color_continuous_scale="RdBu_r",
                title="Feature Correlation Matrix"
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Top correlations
            st.subheader("Strongest Correlations")
            if corr_df is not None:
                top_corr = corr_df.head(10)
                st.dataframe(top_corr)
        except Exception as e:
            st.error(f"Error generating correlation analysis: {e}")

# Tab 4: Predictions
with tab4:
    st.header("🤖 Model Predictions")
    
    # Model parameters
    st.subheader("Model Configuration")
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    with pred_col1:
        sequence_length = st.slider("Sequence Length", 5, 30, 15)
        epochs = st.slider("Epochs", 10, 100, 20)
    
    with pred_col2:
        hidden_dim = st.slider("Hidden Dimension", 32, 256, 128)
        num_layers = st.slider("Number of Layers", 1, 5, 3)
    
    with pred_col3:
        batch_size = st.slider("Batch Size", 16, 128, 32)
        mc_samples = st.slider("MC Samples", 10, 100, 30)
    
    # Prediction button
    if st.button("🚀 Run Prediction"):
        if tensor_data is not None:
            with st.spinner("Training model and generating predictions..."):
                try:
                    dates, predictions, actuals, std_devs = mlt.parallelized_rolling_window_prediction_for_financial_data2(
                        tensor_data,
                        target_variable='Open',
                        sequence_length=sequence_length,
                        epochs=epochs,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        batch_size=batch_size,
                        mc_samples=mc_samples,
                        use_features=True
                    )
                    
                    # Plot predictions
                    fig_pred = go.Figure()
                    
                    # Actual values
                    fig_pred.add_trace(go.Scatter(
                        x=dates,
                        y=actuals,
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue')
                    ))
                    
                    # Predictions
                    fig_pred.add_trace(go.Scatter(
                        x=dates,
                        y=predictions,
                        mode='lines',
                        name='Predicted',
                        line=dict(color='red')
                    ))
                    
                    # Uncertainty bands
                    upper_bound = predictions + std_devs
                    lower_bound = predictions - std_devs
                    
                    fig_pred.add_trace(go.Scatter(
                        x=dates,
                        y=upper_bound,
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        showlegend=False
                    ))
                    
                    fig_pred.add_trace(go.Scatter(
                        x=dates,
                        y=lower_bound,
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        name='Uncertainty',
                        fillcolor='rgba(255,0,0,0.2)'
                    ))
                    
                    fig_pred.update_layout(
                        title="Stock Price Predictions with Uncertainty",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        height=600
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Performance metrics
                    mse = np.mean((predictions - actuals) ** 2)
                    mae = np.mean(np.abs(predictions - actuals))
                    
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("Mean Squared Error", f"{mse:.4f}")
                    with metric_col2:
                        st.metric("Mean Absolute Error", f"{mae:.4f}")
                        
                except Exception as e:
                    st.error(f"Error running prediction: {e}")
        else:
            st.warning("Please load data first.")

# Tab 5: Sentiment Analysis
with tab5:
    st.header("📰 Sentiment Analysis")
    
    sentiment_data = load_sentiment_data(selected_stock)
    
    if sentiment_data is not None:
        # Sentiment overview
        st.subheader("Sentiment Overview")
        
        # Sentiment metrics
        avg_sentiment = sentiment_data['sentiment_score'].mean()
        sentiment_std = sentiment_data['sentiment_score'].std()
        
        sent_col1, sent_col2, sent_col3 = st.columns(3)
        with sent_col1:
            st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
        with sent_col2:
            st.metric("Sentiment Volatility", f"{sentiment_std:.3f}")
        with sent_col3:
            positive_pct = (sentiment_data['sentiment_score'] > 0).mean() * 100
            st.metric("Positive News %", f"{positive_pct:.1f}%")
        
        # Sentiment over time
        date_col = None
        for col in sentiment_data.columns:
            if 'date' in col.lower():
                date_col = col
                break
        
        if date_col is not None:
            try:
                sentiment_dates = pd.to_datetime(sentiment_data[date_col])
                fig_sent = px.line(x=sentiment_dates, y=sentiment_data['sentiment_score'], 
                                 title="Sentiment Score Over Time")
                fig_sent.update_xaxes(title="Date")
                fig_sent.update_yaxes(title="Sentiment Score")
                st.plotly_chart(fig_sent, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting sentiment over time: {e}")
        else:
            st.warning("Date column not found in sentiment data")
        
        # Sentiment distribution
        fig_dist = px.histogram(sentiment_data, x='sentiment_score', 
                               title="Distribution of Sentiment Scores")
        st.plotly_chart(fig_dist, use_container_width=True)
        
    else:
        st.warning(f"No sentiment data available for {selected_stock_name}")

# Tab 6: Model Configuration
with tab6:
    st.header("⚙️ Advanced Model Configuration")
    
    # Model architecture settings
    st.subheader("Model Architecture")
    
    arch_col1, arch_col2 = st.columns(2)
    with arch_col1:
        st.write("**LSTM Parameters:**")
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
        learning_rate = st.number_input("Learning Rate", value=0.001, format="%.4f")
    
    with arch_col2:
        st.write("**Training Parameters:**")
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
        early_stopping = st.checkbox("Early Stopping", value=True)
    
    # Feature selection
    st.subheader("Feature Selection")
    if loaded_data is not None:
        available_features = []
        for dataset_name, df in loaded_data.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            available_features.extend([f"{dataset_name}:{col}" for col in numeric_cols])
        
        selected_features = st.multiselect(
            "Select Features for Training:",
            available_features,
            default=available_features[:5] if len(available_features) >= 5 else available_features
        )
    
    # Model management
    st.subheader("Model Management")
    model_col1, model_col2 = st.columns(2)
    
    with model_col1:
        if st.button("💾 Save Model Configuration"):
            st.success("Model configuration saved!")
    
    with model_col2:
        if st.button("📁 Load Model Configuration"):
            st.success("Model configuration loaded!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Machine Learning Stock Prediction Dashboard")
