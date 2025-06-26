"""
Machine Learning Stock Prediction Dashboard
Built with Streamlit for interactive visualization and model management
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import importlib

# Add src directory to Python path
sys.path.append('./src')

# Import your custom modules - following machine_learning_main.ipynb structure
try:
    import machine_learning_data as mld
    import machine_learning_plotting as mlp
    import machine_learning_training as mlt
    import machine_learning_dataloading as mldl
    from download_csv import *
    from machine_learning_BERT_articles import *
    from create_tensor import *
    
    # Reload modules to get latest changes (following notebook pattern)
    importlib.reload(mlp)
    importlib.reload(mld)
    importlib.reload(mlt)
    importlib.reload(mldl)
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

# Load data function with caching - following machine_learning_main.ipynb pattern
@st.cache_data
def load_financial_data():
    """Load and cache financial data using the pattern from machine_learning_main.ipynb"""
    try:
        # Following the exact pattern from machine_learning_main.ipynb
        # Create tensor from CSV files - this loads and aligns all data
        tensor_data, loaded_data = mldl.create_tensor_from_csvs([
            './data/financial_data.csv', 
            './data/gdp_data.csv', 
            './data/interest_rate_data.csv',
            './data/inflation_data.csv',
            './data/unemployment_rate_data.csv',
            './data/SP500_sentiment_gpu_parallel_filtered.csv'
        ])
        
        return tensor_data, loaded_data
        
    except Exception as e:
        st.error(f"Error in load_financial_data: {e}")
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
                df_type = type(df).__name__
                if hasattr(df, 'shape'):
                    st.write(f"- {key}: {df_type} {df.shape}")
                    if hasattr(df, 'columns'):
                        st.write(f"  Columns: {list(df.columns)}")
                    else:
                        st.write(f"  No columns attribute (Series)")
                else:
                    st.write(f"- {key}: {df_type} (no shape attribute)")
        else:
            st.write("No data loaded")
    
    if st.button("Test Correlation Function"):
        tensor_data, loaded_data = load_financial_data()
        if loaded_data:
            st.write("**Testing correlation function:**")
            try:
                import machine_learning_plotting as mlp
                corr_matrix, found_vars, corr_df = mlp.create_half_correlation_plot3(loaded_data, plot=False, save=False)
                st.write("✅ Correlation function worked")
                st.write(f"Matrix shape: {corr_matrix.shape if hasattr(corr_matrix, 'shape') else 'No shape'}")
                st.write(f"Found variables: {len(found_vars) if found_vars else 0}")
            except Exception as e:
                st.write(f"❌ Correlation function failed: {e}")
        else:
            st.write("No data to test")

# Helper function to display matplotlib plots in Streamlit
def display_matplotlib_plot(fig):
    """Display a matplotlib figure in Streamlit"""
    st.pyplot(fig, clear_figure=True)

# Helper function to capture plot from mlp functions for display
def capture_mlp_plot(plot_func, *args, **kwargs):
    """Capture matplotlib plot from mlp functions and display in Streamlit"""
    # Temporarily disable the plot parameter and capture the figure  
    if 'plot' in kwargs:
        kwargs['plot'] = False
    
    # Call the function to get data, then create plot manually if needed
    result = plot_func(*args, **kwargs)
    return result

# Enhanced matplotlib integration for Streamlit
def streamlit_show_mlp_plots(loaded_data):
    """
    Display plots from machine_learning_plotting functions in Streamlit
    This handles the matplotlib to streamlit conversion
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Temporarily store the current plt state
        import matplotlib.pyplot as plt
        
        # Clear any existing plots
        plt.close('all')
        
        # Call the plotting function - it will create figures
        st.write("Creating financial data plots...")
        
        # Patch plt.show() to capture figures instead of displaying them
        original_show = plt.show
        captured_figures = []
        
        def capture_show(block=None):
            # Instead of showing, save the current figure
            current_fig = plt.gcf()
            if current_fig.get_axes():  # Only save if figure has content
                captured_figures.append(current_fig)
            # Create a new figure for the next plot
            plt.figure(figsize=(14, 10))
        
        # Replace plt.show temporarily
        plt.show = capture_show
        
        try:
            # Now call the mlp function - it will use our custom show function
            mlp.plot_financial_data_from_tensor(loaded_data, plot=True)
            
            # Display all captured figures in Streamlit
            for i, fig in enumerate(captured_figures):
                st.pyplot(fig, clear_figure=True)
                
        finally:
            # Restore original plt.show
            plt.show = original_show
            plt.close('all')
            
        if not captured_figures:
            st.warning("No plots were generated by the plotting function")
            
    except Exception as e:
        st.error(f"Error in streamlit_show_mlp_plots: {e}")
        st.code(str(e))

# Tab 1: Data Overview
with tab1:
    st.header("📊 Financial Data Overview")
    
    # Load data with detailed error reporting - following machine_learning_main.ipynb pattern
    try:
        st.write("🔄 Loading data using tensor creation...")
        tensor_data, loaded_data = load_financial_data()
        
        if loaded_data is None:
            st.error("❌ Data loading returned None")
            st.write("**Possible solutions:**")
            st.write("1. Ensure data files exist in the `./data/` folder")
            st.write("2. Run data download scripts if needed")
            st.write("3. Check if tensor creation functions work correctly")
            st.stop()
        
        st.success(f"✅ Successfully loaded tensor data with shape: {tensor_data.shape if tensor_data is not None else 'N/A'}")
        
        # Debug information
        with st.expander("� Data Loading Debug Info", expanded=False):
            st.write(f"**Tensor data shape:** {tensor_data.shape if tensor_data is not None else 'None'}")
            st.write(f"**Loaded data type:** {type(loaded_data)}")
            if hasattr(loaded_data, 'columns'):
                st.write(f"**Data columns:** {list(loaded_data.columns)}")
                st.write(f"**Data shape:** {loaded_data.shape}")
            elif isinstance(loaded_data, dict):
                st.write(f"**Data keys:** {list(loaded_data.keys())}")
            
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.write("**Debug info:**")
        st.code(str(e))
        st.stop()
    
    # Display financial data plots using mlp.plot_financial_data_from_tensor
    # This follows the exact pattern from machine_learning_main.ipynb
    if loaded_data is not None:
        st.subheader("📈 Financial Data Visualization")
        
        try:
            # Following notebook pattern: mlp.plot_financial_data_from_tensor(loaded_data, plot=True)
            # Use our enhanced matplotlib integration
            streamlit_show_mlp_plots(loaded_data)
            
        except Exception as e:
            st.error(f"Error creating plots with mlp functions: {e}")
            st.write(f"Error details: {str(e)}")
            
            # Fallback to basic information display
            st.subheader("📋 Data Information (Fallback)")
            if isinstance(loaded_data, dict):
                for key, data in loaded_data.items():
                    st.write(f"**{key}:**")
                    if hasattr(data, 'shape'):
                        st.write(f"  - Shape: {data.shape}")
                    if hasattr(data, 'columns'):
                        st.write(f"  - Columns: {list(data.columns)}")
                    if hasattr(data, 'dtypes'):
                        st.write(f"  - Data types: {dict(data.dtypes)}")
            elif hasattr(loaded_data, 'shape'):
                st.write(f"**Data shape:** {loaded_data.shape}")
                st.write(f"**Columns:** {list(loaded_data.columns) if hasattr(loaded_data, 'columns') else 'N/A'}")
        
        # Display data summary
        st.subheader("📋 Data Summary")
        
        if hasattr(loaded_data, 'describe'):
            # If loaded_data is a DataFrame
            st.write("**Statistical Summary:**")
            st.dataframe(loaded_data.describe())
            
            # Show metrics if we have financial columns
            financial_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [col for col in financial_cols if col in loaded_data.columns]
            
            if available_cols:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if 'Close' in loaded_data.columns:
                        current_price = loaded_data['Close'].iloc[-1]
                        st.metric("Latest Close Price", f"${current_price:.2f}")
                
                with col2:
                    if 'Close' in loaded_data.columns and len(loaded_data) > 1:
                        current_price = loaded_data['Close'].iloc[-1]
                        prev_price = loaded_data['Close'].iloc[-2]
                        price_change = current_price - prev_price
                        pct_change = (price_change / prev_price) * 100
                        st.metric("Daily Change", f"${price_change:.2f}", delta=f"{pct_change:.2f}%")
                
                with col3:
                    if 'Volume' in loaded_data.columns:
                        volume = loaded_data['Volume'].iloc[-1]
                        st.metric("Latest Volume", f"{volume:,.0f}")
                
                with col4:
                    if 'Close' in loaded_data.columns and len(loaded_data) > 20:
                        volatility = loaded_data['Close'].pct_change().std() * np.sqrt(252) * 100
                        st.metric("Annualized Volatility", f"{volatility:.2f}%")
        
        elif isinstance(loaded_data, dict):
            # If loaded_data is a dictionary of datasets
            st.write("**Available Datasets:**")
            for key, data in loaded_data.items():
                if hasattr(data, 'shape'):
                    st.write(f"- **{key}**: {data.shape[0]} rows, {data.shape[1]} columns")
                else:
                    st.write(f"- **{key}**: {type(data)}")
    
    else:
        st.warning("No data available for visualization. Please check data loading.")
        
        # Data file status check
        with st.expander("📁 Data Files Status"):
            data_files = [
                './data/financial_data.csv',
                './data/gdp_data.csv', 
                './data/interest_rate_data.csv',
                './data/inflation_data.csv',
                './data/unemployment_rate_data.csv',
                './data/SP500_sentiment_gpu_parallel_filtered.csv'
            ]
            
            for file_path in data_files:
                if os.path.exists(file_path):
                    st.write(f"✅ {file_path}")
                else:
                    st.write(f"❌ {file_path}")

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
        st.subheader("Financial Data Correlations")
        st.write("Using machine_learning_plotting correlation analysis...")
        
        try:
            # Following the pattern from machine_learning_main.ipynb:
            # corr_matrix, found_vars, corr_df = mlp.create_half_correlation_plot3(loaded_data, plot=True, save=False)
            
            # Use the mlp correlation function but don't plot directly (we'll handle the plot display)
            corr_matrix, found_vars, corr_df = mlp.create_half_correlation_plot3(loaded_data, plot=False, save=False)
            
            st.success(f"✅ Created correlation analysis with {len(found_vars)} variables")
            
            # Display the variables that were found and used
            with st.expander("📋 Variables Used in Correlation Analysis"):
                for var in found_vars:
                    st.write(f"- {var}")
            
            # Display correlation matrix as interactive heatmap
            if corr_matrix is not None and not corr_matrix.empty:
                fig_corr = px.imshow(
                    corr_matrix,
                    labels=dict(x="Variables", y="Variables", color="Correlation"),
                    color_continuous_scale="RdBu_r",
                    title=f"Financial Data Correlation Matrix ({len(found_vars)} variables)",
                    aspect="auto"
                )
                fig_corr.update_layout(
                    height=max(600, len(found_vars) * 30),
                    xaxis={'side': 'bottom'},
                    font=dict(size=10)
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Display correlation results table
            if corr_df is not None and not corr_df.empty:
                st.subheader("📊 Correlation Results")
                st.dataframe(
                    corr_df.style.format({'Correlation': '{:.3f}'}),
                    use_container_width=True
                )
                
                # Top correlations visualization
                if len(corr_df) > 0:
                    st.subheader("🔝 Strongest Correlations")
                    top_n = min(10, len(corr_df))
                    top_corr = corr_df.head(top_n)
                    
                    fig_bar = px.bar(
                        top_corr,
                        x='Correlation',
                        y=top_corr.index,
                        orientation='h',
                        title=f"Top {top_n} Strongest Correlations",
                        labels={'y': 'Variable Pairs', 'Correlation': 'Correlation Coefficient'}
                    )
                    fig_bar.update_layout(height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in correlation analysis: {e}")
            st.write("**Error details:**")
            st.code(str(e))
            
            # Fallback: Basic correlation analysis
            st.subheader("Fallback Correlation Analysis")
            try:
                # Simple correlation for debugging
                if hasattr(loaded_data, 'corr'):
                    # If loaded_data is a DataFrame
                    corr_matrix = loaded_data.corr()
                    fig_corr = px.imshow(
                        corr_matrix,
                        title="Basic Correlation Matrix",
                        color_continuous_scale="RdBu_r"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.write("Cannot perform fallback correlation - data structure not supported")
            except Exception as fallback_error:
                st.error(f"Fallback correlation also failed: {fallback_error}")
    
    else:
        st.warning("No data available for correlation analysis")

# Tab 4: Predictions
with tab4:
    st.header("🤖 LSTM-RNN Predictions")
    
    # Following the exact pattern from machine_learning_main.ipynb
    st.subheader("Model Configuration")
    
    # Model parameters based on the notebook defaults
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    with pred_col1:
        sequence_length = st.slider("Sequence Length", 5, 30, 15, help="Number of time steps to look back")
        epochs = st.slider("Epochs", 10, 100, 20, help="Number of training epochs")
    
    with pred_col2:
        hidden_dim = st.slider("Hidden Dimension", 32, 256, 128, help="LSTM hidden layer size")
        num_layers = st.slider("Number of Layers", 1, 5, 3, help="Number of LSTM layers")
    
    with pred_col3:
        batch_size = st.slider("Batch Size", 16, 128, 32, help="Training batch size")
        mc_samples = st.slider("MC Samples", 10, 100, 30, help="Monte Carlo samples for uncertainty")
    
    # Target variable selection
    target_variable = st.selectbox("Target Variable", ['Open', 'Close', 'High', 'Low'], index=0)
    use_features = st.checkbox("Use Additional Features", value=True, help="Include economic indicators and sentiment")
    
    # Prediction button
    if st.button("🚀 Run LSTM Prediction", type="primary"):
        if tensor_data is not None:
            with st.spinner("Training LSTM model and generating predictions..."):
                try:
                    # Following exact pattern from notebook:
                    # dates, predictions, actuals, std_devs = mlt.parallelized_rolling_window_prediction_for_financial_data2(...)
                    
                    st.info("Training LSTM model with Monte Carlo dropout for uncertainty estimation...")
                    
                    dates, predictions, actuals, std_devs = mlt.parallelized_rolling_window_prediction_for_financial_data2(
                        tensor_data,
                        target_variable=target_variable,
                        sequence_length=sequence_length,
                        epochs=epochs,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        batch_size=batch_size,
                        mc_samples=mc_samples,
                        use_features=use_features
                    )
                    
                    st.success(f"✅ Model training completed! Generated {len(predictions)} predictions.")
                    
                    # Display prediction results
                    st.subheader("📈 Prediction Results")
                    
                    # Plot predictions with uncertainty
                    fig_pred = go.Figure()
                    
                    # Actual values
                    fig_pred.add_trace(go.Scatter(
                        x=dates,
                        y=actuals,
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Predictions
                    fig_pred.add_trace(go.Scatter(
                        x=dates,
                        y=predictions,
                        mode='lines',
                        name='Predicted',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Uncertainty bands (confidence intervals)
                    if std_devs is not None and len(std_devs) > 0:
                        upper_bound = predictions + 2 * std_devs  # 95% confidence interval
                        lower_bound = predictions - 2 * std_devs
                        
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
                            name='95% Confidence Interval',
                            fillcolor='rgba(255,0,0,0.2)'
                        ))
                    
                    fig_pred.update_layout(
                        title=f"LSTM Stock Price Predictions - {target_variable}",
                        xaxis_title="Date",
                        yaxis_title=f"Price ({target_variable})",
                        height=600,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Performance metrics
                    st.subheader("📊 Model Performance")
                    
                    # Calculate metrics
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    mse = mean_squared_error(actuals, predictions)
                    mae = mean_absolute_error(actuals, predictions)
                    r2 = r2_score(actuals, predictions)
                    rmse = np.sqrt(mse)
                    
                    # Display metrics
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric("RMSE", f"{rmse:.3f}")
                    with metric_col2:
                        st.metric("MAE", f"{mae:.3f}")
                    with metric_col3:
                        st.metric("R²", f"{r2:.3f}")
                    with metric_col4:
                        mean_uncertainty = np.mean(std_devs) if std_devs is not None else 0
                        st.metric("Avg Uncertainty", f"{mean_uncertainty:.3f}")
                    
                    # Prediction vs Actual scatter plot
                    st.subheader("📊 Prediction vs Actual")
                    
                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(go.Scatter(
                        x=actuals,
                        y=predictions,
                        mode='markers',
                        name='Predictions',
                        marker=dict(color='red', size=6, opacity=0.6)
                    ))
                    
                    # Perfect prediction line
                    min_val = min(min(actuals), min(predictions))
                    max_val = max(max(actuals), max(predictions))
                    fig_scatter.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='black', dash='dash')
                    ))
                    
                    fig_scatter.update_layout(
                        title="Predicted vs Actual Values",
                        xaxis_title="Actual Values",
                        yaxis_title="Predicted Values",
                        height=500
                    )
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
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
        # Check what columns are available in sentiment data
        available_columns = list(sentiment_data.columns)
        
        # Calculate sentiment score if individual sentiment columns exist
        if 'sentiment_score' in available_columns:
            sentiment_col = 'sentiment_score'
        elif all(col in available_columns for col in ['positive', 'negative', 'neutral']):
            # Calculate composite sentiment score: positive - negative
            sentiment_data['sentiment_score'] = sentiment_data['positive'] - sentiment_data['negative']
            sentiment_col = 'sentiment_score'
        elif 'positive' in available_columns:
            sentiment_col = 'positive'
        else:
            # Find any numeric column that might represent sentiment
            numeric_cols = sentiment_data.select_dtypes(include=[np.number]).columns.tolist()
            sentiment_col = numeric_cols[0] if numeric_cols else None
        
        if sentiment_col is not None:
            # Sentiment overview
            st.subheader("Sentiment Overview")
            
            # Sentiment metrics
            avg_sentiment = sentiment_data[sentiment_col].mean()
            sentiment_std = sentiment_data[sentiment_col].std()
            
            sent_col1, sent_col2, sent_col3 = st.columns(3)
            with sent_col1:
                st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
            with sent_col2:
                st.metric("Sentiment Volatility", f"{sentiment_std:.3f}")
            with sent_col3:
                if sentiment_col == 'sentiment_score' or sentiment_col == 'positive':
                    positive_pct = (sentiment_data[sentiment_col] > 0).mean() * 100
                    st.metric("Positive %", f"{positive_pct:.1f}%")
                else:
                    st.metric("Data Points", f"{len(sentiment_data)}")
            
            # Show column breakdown if available
            if all(col in available_columns for col in ['positive', 'negative', 'neutral']):
                st.subheader("Sentiment Breakdown")
                breakdown_cols = st.columns(3)
                with breakdown_cols[0]:
                    avg_pos = sentiment_data['positive'].mean()
                    st.metric("Avg Positive", f"{avg_pos:.3f}")
                with breakdown_cols[1]:
                    avg_neu = sentiment_data['neutral'].mean()
                    st.metric("Avg Neutral", f"{avg_neu:.3f}")
                with breakdown_cols[2]:
                    avg_neg = sentiment_data['negative'].mean()
                    st.metric("Avg Negative", f"{avg_neg:.3f}")
            
            # Sentiment over time
            date_col = None
            for col in sentiment_data.columns:
                if 'date' in col.lower():
                    date_col = col
                    break
            
            if date_col is not None:
                try:
                    sentiment_dates = pd.to_datetime(sentiment_data[date_col])
                    
                    # Plot main sentiment metric
                    fig_sent = px.line(x=sentiment_dates, y=sentiment_data[sentiment_col], 
                                     title=f"{sentiment_col.title()} Over Time")
                    fig_sent.update_xaxes(title="Date")
                    fig_sent.update_yaxes(title=sentiment_col.title())
                    st.plotly_chart(fig_sent, use_container_width=True)
                    
                    # If we have breakdown data, show it too
                    if all(col in available_columns for col in ['positive', 'negative', 'neutral']):
                        fig_breakdown = go.Figure()
                        fig_breakdown.add_trace(go.Scatter(
                            x=sentiment_dates, y=sentiment_data['positive'],
                            mode='lines', name='Positive', line=dict(color='green')
                        ))
                        fig_breakdown.add_trace(go.Scatter(
                            x=sentiment_dates, y=sentiment_data['negative'],
                            mode='lines', name='Negative', line=dict(color='red')
                        ))
                        fig_breakdown.add_trace(go.Scatter(
                            x=sentiment_dates, y=sentiment_data['neutral'],
                            mode='lines', name='Neutral', line=dict(color='gray')
                        ))
                        fig_breakdown.update_layout(
                            title="Sentiment Components Over Time",
                            xaxis_title="Date",
                            yaxis_title="Sentiment Score",
                            height=400
                        )
                        st.plotly_chart(fig_breakdown, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error plotting sentiment over time: {e}")
            else:
                st.warning("Date column not found in sentiment data")
            
            # Sentiment distribution
            fig_dist = px.histogram(sentiment_data, x=sentiment_col, 
                                   title=f"Distribution of {sentiment_col.title()}")
            st.plotly_chart(fig_dist, use_container_width=True)
            
        else:
            st.error("No suitable sentiment column found in the data")
            st.write("Available columns:", available_columns)
        
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
            # Ensure df is a DataFrame
            if isinstance(df, pd.Series):
                df = df.to_frame()
            
            # Check if df is a DataFrame and has the select_dtypes method
            if hasattr(df, 'select_dtypes'):
                try:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    available_features.extend([f"{dataset_name}:{col}" for col in numeric_cols])
                except Exception as e:
                    st.warning(f"Could not process {dataset_name}: {e}")
            else:
                st.warning(f"{dataset_name} is not a valid DataFrame: {type(df)}")
        
        if available_features:
            selected_features = st.multiselect(
                "Select Features for Training:",
                available_features,
                default=available_features[:5] if len(available_features) >= 5 else available_features
            )
        else:
            st.warning("No numeric features found in the loaded data.")
    
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
