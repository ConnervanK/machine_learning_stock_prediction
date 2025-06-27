
"""
Financial data plotting functions
"""


def load_and_plot_temporal_data(folder_path='data'):
    """
    Load temporal data files from a folder and plot their values over time
    
    Args:
        folder_path: Path to the folder containing temporal data files
    
    Returns:
        Dictionary containing loaded dataframes and their date columns
    """

    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime

    print(f"Scanning folder: {folder_path}")
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found")
        return {}
    
    # List all files in the directory
    files = os.listdir(folder_path)
    
    # Filter for common data file types
    data_files = [f for f in files if f.endswith(('.csv', '.xlsx', '.xls', '.txt', '.json'))]
    
    if not data_files:
        print(f"No data files found in '{folder_path}'")
        return {}
    
    print(f"Found {len(data_files)} data file(s): {', '.join(data_files)}")
    
    # Dictionary to store loaded data
    loaded_data = {}
    
    # Process each file
    for filename in data_files:
        file_path = os.path.join(folder_path, filename)
        print(f"\nProcessing file: {filename}")
        
        try:
            # Load data based on file extension
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif filename.endswith('.txt'):
                # Try different delimiters
                for delimiter in [',', '\t', ';', '|', ' ']:
                    try:
                        df = pd.read_csv(file_path, delimiter=delimiter)
                        if len(df.columns) > 1:  # Found a good delimiter
                            break
                    except:
                        pass
            elif filename.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                print(f"  Skipping unsupported file type: {filename}")
                continue
            
            print(f"  Loaded data with shape: {df.shape}")
            
            # Display column information
            print(f"  Columns: {', '.join(df.columns.tolist())}")
            
            # Try to identify date/time column
            date_col = None
            for col in df.columns:
                # Check common date column names
                if col.lower() in ['date', 'time', 'timestamp', 'datetime', 'period']:
                    date_col = col
                    break
                
                # If not found by name, check data types
                if date_col is None:
                    for col in df.columns:
                        try:
                            # Check if column can be converted to datetime
                            pd.to_datetime(df[col])
                            date_col = col
                            print(f"  Identified date column: {date_col}")
                            break
                        except:
                            continue
            
            # If date column found, convert to datetime
            if date_col:
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    print(f"  Date range: {df[date_col].min()} to {df[date_col].max()}")
                except Exception as e:
                    print(f"  Warning: Could not convert {date_col} to datetime. Error: {str(e)}")
            else:
                print("  No date column identified. Will use index for plotting.")
            
            # Store the dataframe and identified date column
            loaded_data[filename] = {
                'df': df,
                'date_col': date_col
            }
            
            # Plot the data over time
            plot_temporal_data(df, date_col, filename)
            
        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
    
    return loaded_data

def load_and_plot_financial_data2(data_dir='data', plot=True):
    """
    Load financial data from CSV files and create organized plots:
    - Stock data: Open, Close, High, Low, Volume in subplots for each stock
    - Financial indicators: Interest rates, GDP, etc. in separate subplots
    
    Args:
        data_dir (str): Directory containing CSV files with financial data
        plot (bool): Whether to generate plots
        
    Returns:
        dict: Dictionary of dataframes with aligned dates
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    import numpy as np
    from datetime import datetime
    import matplotlib.gridspec as gridspec
    
    print(f"Loading data from {data_dir}...")
    
    # Get all CSV files in the directory
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return {}
    
    # Load all CSV files into dataframes
    dataframes = {}
    all_dates = set()
    
    for file in csv_files:
        filename = os.path.basename(file)
        name = os.path.splitext(filename)[0]
        
        try:
            # Load the dataframe
            df = pd.read_csv(file, parse_dates=['date'])
            df.set_index('date', inplace=True)
            
            # Store the dataframe and collect dates
            dataframes[name] = df
            all_dates.update(df.index)
            print(f"Loaded {name} with {len(df)} records")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Only generate plots if requested
    if plot:
        # Function to format dollar values on y-axis
        def dollar_formatter(x, pos):
            if x >= 1e9:
                return f'${x/1e9:.1f}B'
            elif x >= 1e6:
                return f'${x/1e6:.1f}M'
            elif x >= 1e3:
                return f'${x/1e3:.1f}K'
            else:
                return f'${x:.2f}'
        
        # Function to format volume values
        def volume_formatter(x, pos):
            if x >= 1e9:
                return f'{x/1e9:.1f}B'
            elif x >= 1e6:
                return f'{x/1e6:.1f}M'
            elif x >= 1e3:
                return f'{x/1e3:.1f}K'
            else:
                return f'{int(x)}'
        
        # Identify stock dataframes vs. financial indicator dataframes
        stock_dfs = {}
        indicator_dfs = {}
        
        # Stock data typically has these columns
        stock_columns = ['Open', 'Close', 'High', 'Low', 'Volume']
        
        for name, df in dataframes.items():
            if any(col in df.columns for col in stock_columns):
                stock_dfs[name] = df
            else:
                indicator_dfs[name] = df
        
        print(f"Found {len(stock_dfs)} stock datasets and {len(indicator_dfs)} financial indicator datasets")
        
        # Create stock plots (one figure per stock)
        for stock_name, df in stock_dfs.items():
            fig = plt.figure(figsize=(14, 16))
            
            # Create grid with different heights for subplots
            gs = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 4])
            
            # Create a title for the entire figure
            fig.suptitle(f"{stock_name.replace('_', ' ').title()} Stock Data", 
                         fontsize=20, fontweight='bold', y=0.98)
            
            # Plot 1: Open prices
            ax1 = plt.subplot(gs[0])
            if 'Open' in df.columns:
                ax1.plot(df.index, df['Open'], color='blue', linewidth=1.5)
                ax1.set_title('Opening Price', fontsize=14)
                ax1.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
                ax1.grid(True, alpha=0.3)
                # Only show year and month on x-axis
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Close prices
            ax2 = plt.subplot(gs[1], sharex=ax1)
            if 'Close' in df.columns:
                ax2.plot(df.index, df['Close'], color='green', linewidth=1.5)
                ax2.set_title('Closing Price', fontsize=14)
                ax2.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
                ax2.grid(True, alpha=0.3)
                # Hide x-axis labels for middle plots
                plt.setp(ax2.get_xticklabels(), visible=False)
            
            # Plot 3: High and Low prices
            ax3 = plt.subplot(gs[2], sharex=ax1)
            if 'High' in df.columns and 'Low' in df.columns:
                ax3.plot(df.index, df['High'], color='red', linewidth=1.5, label='High')
                ax3.plot(df.index, df['Low'], color='purple', linewidth=1.5, label='Low')
                ax3.set_title('High and Low Prices', fontsize=14)
                ax3.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
                ax3.grid(True, alpha=0.3)
                ax3.legend()
                plt.setp(ax3.get_xticklabels(), visible=False)
            
            # Plot 4: Price Range (Candlestick-like visualization)
            ax4 = plt.subplot(gs[3], sharex=ax1)
            if all(col in df.columns for col in ['Open', 'Close', 'High', 'Low']):
                # Fill between High and Low
                ax4.fill_between(df.index, df['High'], df['Low'], color='lightgray', alpha=0.5, label='Range')
                # Plot lines for Open and Close
                ax4.plot(df.index, df['Open'], color='blue', alpha=0.7, linewidth=1, label='Open')
                ax4.plot(df.index, df['Close'], color='green', alpha=0.7, linewidth=1, label='Close')
                ax4.set_title('Daily Price Range', fontsize=14)
                ax4.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
                ax4.grid(True, alpha=0.3)
                ax4.legend()
                plt.setp(ax4.get_xticklabels(), visible=False)
            
            # Plot 5: Volume
            ax5 = plt.subplot(gs[4], sharex=ax1)
            if 'Volume' in df.columns:
                # Plot volume as a bar chart
                ax5.bar(df.index, df['Volume'], color='darkblue', alpha=0.7, width=2)
                ax5.set_title('Trading Volume', fontsize=14)
                ax5.yaxis.set_major_formatter(FuncFormatter(volume_formatter))
                ax5.grid(True, alpha=0.3)
                # Show dates on the bottom subplot
                ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax5.tick_params(axis='x', rotation=45)
            
            # Add some vertical spacing between subplots
            plt.subplots_adjust(hspace=0.3)
            
            # Format x-axis dates for better display
            # Set major ticks every 3 months
            ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            # Set minor ticks every month
            ax5.xaxis.set_minor_locator(mdates.MonthLocator())
            
            # Adjust layout and save
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            # plt.savefig(f"{stock_name}_detailed_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # Create financial indicators plot
        if indicator_dfs:
            n_indicators = len(indicator_dfs)
            fig, axes = plt.subplots(n_indicators, 1, figsize=(14, 4 * n_indicators), sharex=True)
            
            # Handle case with only one indicator
            if n_indicators == 1:
                axes = [axes]
            
            fig.suptitle("Economic and Financial Indicators", fontsize=20, fontweight='bold', y=0.98)
            
            for i, (name, df) in enumerate(indicator_dfs.items()):
                ax = axes[i]
                
                # For each column in the dataframe (usually just one)
                for col in df.columns:
                    ax.plot(df.index, df[col], linewidth=2)
                    
                ax.set_title(f"{name.replace('_', ' ').title()}", fontsize=14)
                
                # Format y-axis based on the indicator type
                if 'rate' in name.lower() or 'interest' in name.lower():
                    # For rates, use percentage
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2f}%"))
                elif 'gdp' in name.lower() or 'economic' in name.lower():
                    # For GDP and economic indicators, use dollars
                    ax.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
                
                ax.grid(True, alpha=0.3)
                
                # Only show x-axis labels on the bottom plot
                if i < n_indicators - 1:
                    plt.setp(ax.get_xticklabels(), visible=False)
            
            # Format x-axis dates for better display on the bottom plot
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            axes[-1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            # plt.savefig("economic_indicators.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    print(f"Loaded {len(dataframes)} datasets")
    return dataframes


def plot_temporal_data(df, date_col=None, filename=''):
    """
    Plot all numerical columns in the dataframe over time
    
    Args:
        df: Pandas DataFrame containing temporal data
        date_col: Column name containing date values (if any)
        filename: Name of the file (for plot title)
    """
    
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Get numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("  No numeric columns found for plotting")
        return
    
    print(f"  Plotting {len(numeric_cols)} numerical columns")
    
    # Create one plot per numerical column for clarity
    for col in numeric_cols:
        plt.figure(figsize=(12, 6))
        
        if date_col:
            plt.plot(df[date_col], df[col], marker='.', linestyle='-', markersize=3)
            plt.xlabel('Date')
        else:
            plt.plot(df[col], marker='.', linestyle='-', markersize=3)
            plt.xlabel('Index')
            
        plt.ylabel(col)
        plt.title(f'{col} over time - {filename}')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        if date_col:
            plt.gcf().autofmt_xdate()
            
        plt.tight_layout()
        plt.show()
    
    # Create a combined plot with all numeric columns
    plt.figure(figsize=(14, 8))
    
    for col in numeric_cols:
        if date_col:
            plt.plot(df[date_col], df[col], label=col)
        else:
            plt.plot(df[col], label=col)
    
    if date_col:
        plt.xlabel('Date')
        plt.gcf().autofmt_xdate()
    else:
        plt.xlabel('Index')
        
    plt.ylabel('Value')
    plt.title(f'All numeric values over time - {filename}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_specific_correlation_plot(loaded_data):
    """
    Create a correlation plot between specific economic and market variables:
    unemployment rate, interest rate, inflation, GDP, open values, and volume values
    
    Args:
        loaded_data: Dictionary containing loaded dataframes from load_and_plot_temporal_data
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    print("Creating correlation plot for specific economic and market variables...")
    
    # Variables we're looking for
    target_variables = ['unemployment', 'interest', 'inflation', 'gdp', 'open', 'volume']
    
    # Dictionary to store found variables
    found_vars = {}
    
    # Search for these variables in all loaded dataframes
    for filename, data_dict in loaded_data.items():
        df = data_dict['df']
        
        # Check each column to find our target variables
        for col in df.columns:
            col_lower = col.lower()
            
            # Check if column name contains any of our target variable names
            for var in target_variables:
                if var in col_lower:
                    # Store the variable name and its data
                    var_key = var.capitalize()
                    if col_lower.startswith('unemployment'):
                        var_key = 'Unemployment'
                    elif col_lower.startswith('interest'):
                        var_key = 'Interest_Rate'
                    elif col_lower.startswith('inflation'):
                        var_key = 'Inflation'
                    elif col_lower.startswith('gdp'):
                        var_key = 'GDP'
                    elif 'open' in col_lower:
                        var_key = 'Open'
                    elif 'volume' in col_lower:
                        var_key = 'Volume'
                    
                    # Store the series and its source
                    found_vars[var_key] = {
                        'data': df[col],
                        'source': filename,
                        'column': col
                    }
                    print(f"Found {var_key} in {filename}, column: {col}")
    
    # Check if we found enough variables to create a correlation plot
    if len(found_vars) < 2:
        print("Not enough variables found for a correlation plot. Need at least 2.")
        return
    
    # Create a new dataframe with our found variables
    corr_df = pd.DataFrame()
    for var_name, var_dict in found_vars.items():
        corr_df[var_name] = var_dict['data']
    
    # Handle missing values (if any)
    if corr_df.isnull().values.any():
        print("Warning: Data contains missing values. Using pairwise correlations.")
        corr_matrix = corr_df.corr(method='pearson', min_periods=1)
    else:
        corr_matrix = corr_df.corr()
    
    # Print the correlation matrix
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(2))
    
    # Create a heatmap
    plt.figure(figsize=(10, 8))
    
    # Use a mask for the upper triangle to make the plot cleaner
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot the heatmap
    import seaborn as sns
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
               mask=mask, vmin=-1, vmax=1, center=0, 
               square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title('Correlation Between Economic and Market Variables', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Also create a scatter plot matrix for more detailed view of relationships
    print("\nCreating scatter plot matrix...")
    if len(found_vars) <= 6:  # Only do this for a reasonable number of variables
        sns.set(style="ticks")
        sns.pairplot(corr_df, diag_kind="kde")
        plt.suptitle('Scatter Plot Matrix of Economic and Market Variables', 
                    y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()
        
    return corr_matrix, found_vars, corr_df

def create_half_correlation_plot(temporal_data, figsize=(14, 12), plot=True, save=True, save_path="images/financial_correlation_analysis.png"):
    """
    Create an enhanced half correlation plot showing stock closing prices and volumes
    with options to control plotting and saving behavior.
    
    Args:
        temporal_data: Dictionary of dataframes from load_and_plot_financial_data
        figsize: Figure size tuple (width, height)
        plot: Whether to display the plot (default: True)
        save: Whether to save the plot to a file (default: True)
        save_path: Path where to save the plot (default: "images/financial_correlation_analysis.png")
        
    Returns:
        corr_matrix: Correlation matrix
        found_vars: List of variables that were used
        corr_df: DataFrame containing the correlation values
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    
    print("Creating enhanced half correlation plot for stock closing prices and volumes...")
    
    # Combine relevant data into a single dataframe
    combined_df = pd.DataFrame()
    found_vars = []
    
    # Extract only closing prices and volumes from stock dataframes
    for dataset_name, df in temporal_data.items():
        # Check if this is a stock dataframe
        if 'Close' in df.columns:
            # Add closing price
            var_name = f"{dataset_name}_Close"
            combined_df[var_name] = df['Close']
            found_vars.append(var_name)
            
            # Add volume if available
            if 'Volume' in df.columns:
                var_name = f"{dataset_name}_Volume"
                combined_df[var_name] = df['Volume']
                found_vars.append(var_name)
        
        # Add other financial indicators (non-stock data)
        elif len(df.columns) == 1:  # Most indicators have only one column
            var_name = dataset_name
            combined_df[var_name] = df.iloc[:, 0]  # Get first column
            found_vars.append(var_name)
    
    # Drop rows with any NaN values for correlation calculation
    combined_df_dropna = combined_df.dropna()
    
    print(f"Selected {len(found_vars)} variables with {len(combined_df_dropna)} complete rows for correlation analysis")
    
    if len(combined_df_dropna) < 5:
        print("Warning: Very few complete rows for correlation analysis. Results may be unreliable.")
    
    # Calculate correlation matrix
    corr_matrix = combined_df_dropna.corr()
    
    if plot or save:
        # Create a mask for the upper triangle and diagonal
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create custom colormap for better visualization
        colors = ["#3498db", "#f5f5f5", "#e74c3c"]
        custom_cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
        
        # Create figure with extra padding at the bottom for colorbar
        fig = plt.figure(figsize=figsize)
        
        # Set seaborn style for a cleaner look
        sns.set_style("white")
        
        # Create a larger bottom margin to accommodate the colorbar
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
        
        # Create the main axes for the heatmap
        ax = plt.subplot(111)
        
        # Draw the heatmap with the mask and improved aesthetics
        # Place colorbar at the bottom to avoid overlap with labels
        cbar_kws = {
            "shrink": 0.8,
            "label": "Correlation Coefficient",
            "orientation": "horizontal",
            "pad": 0.2,  # More padding to separate from the heatmap
            "location": "bottom"  # Place colorbar at bottom
        }
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=custom_cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            linewidths=0.8,
            cbar_kws=cbar_kws,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 10},
            ax=ax
        )
        
        # Improve the appearance of variable names
        def clean_var_name(name):
            # Beautify variable names
            name = name.replace('_', ' ')
            # Capitalize each word
            return ' '.join(word.capitalize() for word in name.split())
        
        # Clean up the axis labels
        plt.xticks(np.arange(len(corr_matrix.columns)) + 0.5, 
                  [clean_var_name(col) for col in corr_matrix.columns], 
                  rotation=45, 
                  ha='right', 
                  fontsize=11)
        
        plt.yticks(np.arange(len(corr_matrix.index)) + 0.5, 
                  [clean_var_name(idx) for idx in corr_matrix.index], 
                  fontsize=11)
        
        # Add grid lines for better visual separation
        ax.set_xticklabels(ax.get_xticklabels(), va='top')
        
        # Single title with better positioning
        plt.suptitle('Financial Variables Correlation Analysis', 
                  fontsize=18, 
                  fontweight='bold',
                  y=0.98)  # Position the title higher
        
        # Add subtitle as text, not as a title to avoid overlap
        fig.text(0.5, 0.94, 
                "Focusing on Stock Closing Prices, Volumes and Economic Indicators", 
                fontsize=13, 
                ha='center', 
                va='center')
        
        # Add explanatory note above the colorbar
        note_text = ("Note: Values indicate Pearson correlation coefficients. "
                    "Closer to +1 (red) indicates strong positive correlation, "
                    "closer to -1 (blue) indicates strong negative correlation.")
        
        # Add note text with better positioning
        fig.text(0.5, 0.02, note_text, ha='center', va='center', fontsize=10, 
                 style='italic', bbox=dict(facecolor='#f9f9f9', alpha=0.5, boxstyle='round,pad=0.5'))
        
        # Add edge highlights to make the heatmap pop
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_edgecolor('#dcdcdc')
        
        # Save the plot with high resolution if requested
        if save:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation plot saved to: {save_path}")
        
        # Show the plot if requested
        if plot:
            plt.show()
        else:
            plt.close(fig)  # Close the figure if not displaying
    
    # Create a DataFrame for easier analysis of correlation values
    corr_df = corr_matrix.copy()
    
    return corr_matrix, found_vars, corr_df


def plot_financial_data_from_tensor(merged_df, plot=True):
    """
    Plot financial data from a merged DataFrame created by create_tensor_from_csvs.
    
    Args:
        merged_df: DataFrame from create_tensor_from_csvs containing financial data
        plot: Whether to generate plots
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    import matplotlib.gridspec as gridspec
    import numpy as np
    
    if not plot:
        return
    
    print("Generating financial data plots from tensor data...")
    
    # Function to format dollar values on y-axis
    def dollar_formatter(x, pos):
        if x >= 1e9:
            return f'${x/1e9:.1f}B'
        elif x >= 1e6:
            return f'${x/1e6:.1f}M'
        elif x >= 1e3:
            return f'${x/1e3:.1f}K'
        else:
            return f'${x:.2f}'
    
    # Function to format volume values
    def volume_formatter(x, pos):
        if x >= 1e9:
            return f'{x/1e9:.1f}B'
        elif x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.1f}K'
        else:
            return f'{int(x)}'
    
    # Function to format percentage values
    def percentage_formatter(x, pos):
        return f'{x:.2f}%'
    
    # Ensure we have a date index
    if 'date' in merged_df.columns:
        merged_df = merged_df.set_index('date')
    
    # Separate stock data and indicator data based on column names
    stock_columns = ['Open', 'Close', 'High', 'Low', 'Volume']
    stock_cols = [col for col in merged_df.columns if col in stock_columns]
    
    # Identify indicator columns by keywords
    interest_cols = [col for col in merged_df.columns if 'interest' in col.lower() or ('rate' in col.lower() and 'unemploy' not in col.lower() and 'inflat' not in col.lower())]
    unemployment_cols = [col for col in merged_df.columns if 'unemploy' in col.lower()]
    inflation_cols = [col for col in merged_df.columns if 'inflat' in col.lower() or 'cpi' in col.lower() or 'consumer price' in col.lower()]
    gdp_cols = [col for col in merged_df.columns if 'gdp' in col.lower()]
    
    # Other columns that don't match specific categories
    other_cols = [col for col in merged_df.columns if 
                 col not in stock_columns and
                 col not in interest_cols and
                 col not in unemployment_cols and
                 col not in inflation_cols and
                 col not in gdp_cols]
    
    # Create indicator dataframes
    indicator_dfs = {}
    
    # Add interest rate data if available
    if interest_cols:
        indicator_dfs['interest_rate'] = merged_df[interest_cols].copy()
    
    # Add unemployment data if available
    if unemployment_cols:
        indicator_dfs['unemployment_rate'] = merged_df[unemployment_cols].copy()
        
    # Add inflation data if available
    if inflation_cols:
        indicator_dfs['inflation_rate'] = merged_df[inflation_cols].copy()
    
    # Add GDP data if available
    if gdp_cols:
        indicator_dfs['gdp'] = merged_df[gdp_cols].copy()
    
    # Add other data if available
    if other_cols:
        indicator_dfs['other_indicators'] = merged_df[other_cols].copy()
    
    # Plot stock data if present
    if stock_cols:
        # We need at least Open or Close column to plot stock data
        if 'Open' in stock_cols or 'Close' in stock_cols:
            stock_df = merged_df[stock_cols]
            
            fig = plt.figure(figsize=(14, 16))
            
            # Create grid with different heights for subplots
            gs = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 4])
            
            # Create a title for the entire figure
            fig.suptitle("Financial Market Data", 
                        fontsize=20, fontweight='bold', y=0.98)
            
            # Plot 1: Open prices
            ax1 = plt.subplot(gs[0])
            if 'Open' in stock_df.columns:
                ax1.plot(stock_df.index, stock_df['Open'], color='blue', linewidth=1.5)
                ax1.set_title('Opening Price', fontsize=14)
                ax1.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
                ax1.grid(True, alpha=0.3)
                # Only show year and month on x-axis
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Close prices
            ax2 = plt.subplot(gs[1], sharex=ax1)
            if 'Close' in stock_df.columns:
                ax2.plot(stock_df.index, stock_df['Close'], color='green', linewidth=1.5)
                ax2.set_title('Closing Price', fontsize=14)
                ax2.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
                ax2.grid(True, alpha=0.3)
                # Hide x-axis labels for middle plots
                plt.setp(ax2.get_xticklabels(), visible=False)
            
            # Plot 3: High and Low prices
            ax3 = plt.subplot(gs[2], sharex=ax1)
            if 'High' in stock_df.columns and 'Low' in stock_df.columns:
                ax3.plot(stock_df.index, stock_df['High'], color='red', linewidth=1.5, label='High')
                ax3.plot(stock_df.index, stock_df['Low'], color='purple', linewidth=1.5, label='Low')
                ax3.set_title('High and Low Prices', fontsize=14)
                ax3.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
                ax3.grid(True, alpha=0.3)
                ax3.legend()
                plt.setp(ax3.get_xticklabels(), visible=False)
            
            # Plot 4: Price Range (Candlestick-like visualization)
            ax4 = plt.subplot(gs[3], sharex=ax1)
            if all(col in stock_df.columns for col in ['Open', 'Close', 'High', 'Low']):
                # Fill between High and Low
                ax4.fill_between(stock_df.index, stock_df['High'], stock_df['Low'], color='lightgray', alpha=0.5, label='Range')
                # Plot lines for Open and Close
                ax4.plot(stock_df.index, stock_df['Open'], color='blue', alpha=0.7, linewidth=1, label='Open')
                ax4.plot(stock_df.index, stock_df['Close'], color='green', alpha=0.7, linewidth=1, label='Close')
                ax4.set_title('Daily Price Range', fontsize=14)
                ax4.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
                ax4.grid(True, alpha=0.3)
                ax4.legend()
                plt.setp(ax4.get_xticklabels(), visible=False)
            
            # Plot 5: Volume
            ax5 = plt.subplot(gs[4], sharex=ax1)
            if 'Volume' in stock_df.columns:
                # Plot volume as a bar chart
                ax5.bar(stock_df.index, stock_df['Volume'], color='darkblue', alpha=0.7, width=2)
                ax5.set_title('Trading Volume', fontsize=14)
                ax5.yaxis.set_major_formatter(FuncFormatter(volume_formatter))
                ax5.grid(True, alpha=0.3)
                # Show dates on the bottom subplot
                ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax5.tick_params(axis='x', rotation=45)
            
            # Add some vertical spacing between subplots
            plt.subplots_adjust(hspace=0.3)
            
            # Format x-axis dates for better display
            # Set major ticks every 3 months
            if hasattr(ax5, 'xaxis'):
                ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                # Set minor ticks every month
                ax5.xaxis.set_minor_locator(mdates.MonthLocator())
            
            # Adjust layout and save
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.savefig("images/financial_market_data.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    # Plot financial indicators - each in its own subplot
    if indicator_dfs:
        n_indicators = len(indicator_dfs)
        
        # Create a figure with enough height for all indicators
        fig, axes = plt.subplots(n_indicators, 1, figsize=(14, 4 * n_indicators))
        
        # Handle case with only one indicator
        if n_indicators == 1:
            axes = [axes]
        
        # Add the title with more space above the top subplot
        fig.suptitle("Economic and Financial Indicators", 
                      fontsize=20, fontweight='bold')
        
        # Add padding at the top to avoid title overlap
        plt.subplots_adjust(top=0.95 - 0.02 * n_indicators)  # Less top space with more indicators
        
        # Define the order of indicators for consistent display
        ordered_keys = ['gdp', 'interest_rate', 'inflation_rate', 'unemployment_rate', 'other_indicators']
        # Filter and sort keys based on what's actually available
        sorted_keys = [key for key in ordered_keys if key in indicator_dfs]
        # Add any keys that might not be in our predefined order
        for key in indicator_dfs.keys():
            if key not in sorted_keys:
                sorted_keys.append(key)
        
        for i, name in enumerate(sorted_keys):
            df = indicator_dfs[name]
            ax = axes[i]
            
            # Choose a color scheme based on the indicator type
            if name == 'inflation_rate':
                colors = ['red', 'darkred', 'firebrick', 'indianred']
            elif name == 'interest_rate':
                colors = ['blue', 'royalblue', 'steelblue', 'deepskyblue']
            elif name == 'unemployment_rate':
                colors = ['purple', 'darkviolet', 'mediumorchid', 'blueviolet']
            elif name == 'gdp':
                colors = ['green', 'darkgreen', 'seagreen', 'mediumseagreen']
            else:
                colors = None  # Use default matplotlib colors
                
            # For each column in the dataframe
            for j, col in enumerate(df.columns):
                color = colors[j % len(colors)] if colors else None
                ax.plot(df.index, df[col], linewidth=2, label=col, color=color)
                
            # Clean up the name for display
            display_name = name.replace('_', ' ').title()
            ax.set_title(display_name, fontsize=14, pad=15)  # More padding for title
            
            # Format y-axis based on the indicator type
            if name == 'inflation_rate':
                # For inflation rates, use percentage with different formatter
                ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
                # Add horizontal line at 0%
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            elif 'interest' in name.lower() or 'rate' in name.lower():
                # For interest rates, use percentage
                ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
            elif 'unemploy' in name.lower():
                # For unemployment rates, use percentage
                ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
            elif 'gdp' in name.lower() or 'economic' in name.lower():
                # For GDP and economic indicators, use dollars
                ax.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
            
            ax.grid(True, alpha=0.3)
            
            # Add legend if we have multiple columns
            if len(df.columns) > 1:
                ax.legend()
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.tick_params(axis='x', rotation=45)
        
        # Adjust spacing between subplots to be larger
        plt.subplots_adjust(hspace=0.5)
        
        plt.tight_layout()
        plt.savefig("images/economic_indicators.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    print("Financial data plots generated successfully")


def create_half_correlation_plot3(loaded_data, figsize=(14, 12), plot=True, save=False, save_path="images/financial_correlation_analysis.png"):
    """
    Create a focused half correlation plot using only Volume, Close price, and economic indicators,
    without including date columns in the correlation calculation.
    
    Args:
        loaded_data: Dictionary of dataframes from load_and_plot_temporal_data
            or DataFrame from create_tensor_from_csvs
        figsize: Figure size tuple (width, height)
        plot: Whether to display the plot (default: True)
        save: Whether to save the plot to a file (default: False)
        save_path: Path where to save the plot (default: "images/financial_correlation_analysis.png")
        
    Returns:
        corr_matrix: Correlation matrix
        found_vars: List of variables that were used
        corr_df: DataFrame containing the correlation values
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    
    print("Creating focused correlation plot for Volume, Close price and economic indicators...")
    
    # Helper function to identify date columns
    def is_date_column(col_series):
        # Check if the column is a datetime type
        if pd.api.types.is_datetime64_any_dtype(col_series):
            return True
        
        # Check if the column name suggests it's a date
        col_name = col_series.name.lower() if hasattr(col_series, 'name') else ''
        date_keywords = ['date', 'time', 'year', 'month', 'day']
        if any(keyword in col_name for keyword in date_keywords):
            return True
            
        # Check if values look like dates
        if isinstance(col_series.iloc[0] if len(col_series) > 0 else None, (str, pd.Timestamp)):
            try:
                pd.to_datetime(col_series.iloc[0])
                return True
            except:
                pass
                
        return False
    
    # Determine if we have a dictionary of dataframes or a single dataframe
    if isinstance(loaded_data, dict):
        # Dictionary format from load_and_plot_temporal_data
        # Create a merged dataframe with selected variables
        focused_df = pd.DataFrame()
        
        # Find stock dataframe(s) that might contain Volume and Close
        stock_dfs = []
        for key, df in loaded_data.items():
            if 'Volume' in df.columns or 'Close' in df.columns:
                # Make a copy to avoid modifying the original
                stock_dfs.append((key, df.copy()))
        
        # Extract the Volume and Close columns from stock dataframes
        for key, df in stock_dfs:
            # Skip date columns
            date_cols = [col for col in df.columns if is_date_column(df[col])]
            for col in date_cols:
                if col in df.columns:
                    df = df.drop(columns=[col])
            
            # Add Volume and Close columns
            if 'Volume' in df.columns:
                focused_df[f'{key}_Volume'] = df['Volume']
            if 'Close' in df.columns:
                focused_df[f'{key}_Close'] = df['Close']
                
        # Add economic indicators
        econ_keys = [k for k in loaded_data.keys() if k not in [key for key, _ in stock_dfs]]
        for key in econ_keys:
            df = loaded_data[key].copy()
            
            # Skip date columns
            date_cols = [col for col in df.columns if is_date_column(df[col])]
            for col in date_cols:
                if col in df.columns:
                    df = df.drop(columns=[col])
                
            # Add remaining economic indicator columns
            for col in df.columns:
                focused_df[f'{key}_{col}'] = df[col]
    
    else:
        # Single dataframe format from create_tensor_from_csvs
        # Copy to avoid modifying original
        df_copy = loaded_data.copy()
        
        # Handle case where 'date' might be in the index
        if pd.api.types.is_datetime64_any_dtype(df_copy.index):
            # Keep the index for alignment but don't include in correlation
            focused_df = pd.DataFrame(index=df_copy.index)
        else:
            focused_df = pd.DataFrame()
            
        # Identify and skip date columns
        date_cols = []
        for col in df_copy.columns:
            if col == 'date' or is_date_column(df_copy[col]):
                date_cols.append(col)
        
        # Drop date columns
        if date_cols:
            df_copy = df_copy.drop(columns=date_cols)
            
        # Extract only Volume and Close columns (if they exist)
        if 'Volume' in df_copy.columns:
            focused_df['Volume'] = df_copy['Volume']
        if 'Close' in df_copy.columns:
            focused_df['Close'] = df_copy['Close']
        
        # Extract economic indicators (all non-stock variables)
        stock_cols = ['Open', 'Close', 'High', 'Low', 'Volume']
        for col in df_copy.columns:
            if col not in stock_cols or col in ['Volume', 'Close']:
                focused_df[col] = df_copy[col]
    
    # Get list of all variables in the focused DataFrame
    found_vars = list(focused_df.columns)
    
    print(f"Selected {len(found_vars)} variables for focused correlation analysis")
    
    # Reset the index if it's a datetime to avoid including it in correlation
    if pd.api.types.is_datetime64_any_dtype(focused_df.index):
        # Store the datetime index temporarily
        datetime_index = focused_df.index
        # Reset index for correlation calculation
        focused_df = focused_df.reset_index(drop=True)
    
    # Drop rows with any NaN values for correlation calculation
    focused_df_dropna = focused_df.dropna()
    
    print(f"Using {len(focused_df_dropna)} complete rows for correlation analysis")
    
    if len(focused_df_dropna) < 5:
        print("Warning: Very few complete rows for correlation analysis. Results may be unreliable.")
    
    # Calculate correlation matrix
    corr_matrix = focused_df_dropna.corr()
    
    if plot or save:
        # Create a mask for the upper triangle and diagonal
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create custom colormap for better visualization
        colors = ["#3498db", "#f5f5f5", "#e74c3c"]
        custom_cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
        
        # Create figure with extra padding at the bottom for colorbar
        fig = plt.figure(figsize=figsize)
        
        # Set seaborn style for a cleaner look
        sns.set_style("white")
        
        # Create a larger bottom margin to accommodate the colorbar
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
        
        # Create the main axes for the heatmap
        ax = plt.subplot(111)
        
        # Draw the heatmap with the mask and improved aesthetics
        # Place colorbar at the bottom to avoid overlap with labels
        cbar_kws = {
            "shrink": 0.8,
            "label": "Correlation Coefficient",
            "orientation": "horizontal",
            "pad": 0.2,  # More padding to separate from the heatmap
            "location": "bottom"  # Place colorbar at bottom
        }
        
        # Handle cases where we have many variables
        if len(corr_matrix) > 20:
            annot = False
            print(f"Warning: Many variables ({len(corr_matrix)}). Disabling annotations for readability.")
        else:
            annot = True
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=custom_cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            linewidths=0.8,
            cbar_kws=cbar_kws,
            annot=annot,
            fmt=".2f",
            annot_kws={"size": 10 if len(corr_matrix) <= 15 else 8},
            ax=ax
        )
        
        # Improve the appearance of variable names
        def clean_var_name(name):
            # Beautify variable names
            name = str(name).replace('_', ' ')
            # Capitalize each word
            return ' '.join(word.capitalize() for word in name.split())
        
        # For better readability with many variables, adjust font size and rotation
        fontsize = 11 if len(corr_matrix) <= 15 else 9 if len(corr_matrix) <= 25 else 7
        rotation = 45 if len(corr_matrix) <= 20 else 90
        
        # Clean up the axis labels
        plt.xticks(np.arange(len(corr_matrix.columns)) + 0.5, 
                  [clean_var_name(col) for col in corr_matrix.columns], 
                  rotation=rotation, 
                  ha='right' if rotation < 90 else 'center', 
                  fontsize=fontsize)
        
        plt.yticks(np.arange(len(corr_matrix.index)) + 0.5, 
                  [clean_var_name(idx) for idx in corr_matrix.index], 
                  fontsize=fontsize)
        
        # Add grid lines for better visual separation
        ax.set_xticklabels(ax.get_xticklabels(), va='top')
        
        # Add title based on data content - no date information
        title = 'Stock Market Volume, Price & Economic Indicators'
        
        # Single title with better positioning - no date range
        plt.suptitle(title, 
                  fontsize=18, 
                  fontweight='bold',
                  y=0.98)  # Position the title higher
        
        # Add subtitle with variable count - no date information
        fig.text(0.5, 0.94, 
                f"Focused Correlation Analysis of {len(corr_matrix)} Key Financial Variables", 
                fontsize=13, 
                ha='center', 
                va='center')
        
        # Add explanatory note above the colorbar
        note_text = ("Note: Values indicate Pearson correlation coefficients. "
                    "Closer to +1 (red) indicates strong positive correlation, "
                    "closer to -1 (blue) indicates strong negative correlation.")
        
        # Add note text with better positioning
        fig.text(0.5, 0.02, note_text, ha='center', va='center', fontsize=10, 
                 style='italic', bbox=dict(facecolor='#f9f9f9', alpha=0.5, boxstyle='round,pad=0.5'))
        
        # Add edge highlights to make the heatmap pop
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_edgecolor('#dcdcdc')
        
        # Save the plot with high resolution if requested
        if save:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation plot saved to: {save_path}")
        
        # Show the plot if requested
        if plot:
            plt.show()
        else:
            plt.close(fig)  # Close the figure if not displaying
    
    # Restore the datetime index for the returned DataFrame if we had one
    if pd.api.types.is_datetime64_any_dtype(loaded_data.index):
        focused_df.index = datetime_index
    
    return corr_matrix, found_vars, focused_df
