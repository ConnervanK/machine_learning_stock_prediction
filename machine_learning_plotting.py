
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