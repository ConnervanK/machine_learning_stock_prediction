"""
Generate financial data
"""

def test():
    """
    Test function to ensure the module is working correctly
    """
    print("Financial data generation module is working correctly.")

def generate_financial_data():

    # from src.create_tensor import extend_monthly_data, create_tensor_from_csvs
    from machine_learning_dataloading import extend_monthly_data, create_tensor_from_csvs
    import download_csv as dcsv
    import os   
    import numpy as np
    from datetime import datetime, timedelta

    # Define the tickers and date range
    # tickers = ['^GSPC', 'EURUSD=X', 'BTC-USD']  # Example tickers: S&P 500, EUR/USD exchange rate, Bitcoin
    tickers = ['^GSPC']  # Example tickers: S&P 500, EUR/USD exchange rate, Bitcoin
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')  # Current date

    # Define the folder to save the CSV files
    folder = 'data'  # Ensure this folder exists or create it
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Download financial data
    dcsv.download_financial_data(tickers, start_date, end_date, 'financial_data.csv', folder)
    dcsv.download_gdp_data(start_date, end_date, 'gdp_data.csv', folder='data')
    dcsv.download_inflation_data(start_date, end_date, 'inflation_data.csv', folder='data')
    dcsv.download_interest_rate_data(start_date, end_date, 'interest_rate_data.csv', folder='data')
    dcsv.download_unemployment_rate_data(start_date, end_date, 'unemployment_rate_data.csv', folder='data')


def prepare_market_data(temporal_data, target_variable='Open'):
    """
    Prepare real market data for LSTM prediction of opening values
    
    Args:
        temporal_data: Dictionary of loaded dataframes from load_and_plot_temporal_data
        target_variable: The target variable to predict (default: 'Open')
    
    Returns:
        dates: Array of dates
        features_data: Processed feature data
        target_idx: Index of the target variable in the processed data
        variable_names: Names of all variables in the processed data
    """
    import pandas as pd
    import numpy as np

    print(f"Preparing market data for predicting '{target_variable}'...")
    
    # Find all available dataframes and columns
    all_variables = {}
    date_columns = {}
    
    for filename, data_dict in temporal_data.items():
        df = data_dict['df']
        date_col = data_dict['date_col']
        
        # If no date column, skip this dataframe
        if date_col is None:
            print(f"  Skipping {filename} - no date column found")
            continue
        
        # Store date column for this dataframe
        date_columns[filename] = date_col
        
        # Get all numeric columns except date
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Store each numeric variable
        for col in numeric_cols:
            # Create a variable name
            var_name = col
            if var_name in all_variables:
                var_name = f"{col}_{filename.split('.')[0]}"
            
            all_variables[var_name] = {
                'data': df[col],
                'date': df[date_col],
                'source': filename
            }
    
    # Check if target variable is available
    target_found = False
    target_name = None
    
    for var_name, var_data in all_variables.items():
        if target_variable.lower() in var_name.lower():
            target_name = var_name
            target_found = True
            print(f"  Found target variable: {var_name}")
            break
    
    if not target_found:
        raise ValueError(f"Target variable '{target_variable}' not found in the data")
    
    # Find common date range for all variables
    min_dates = []
    max_dates = []
    
    for var_name, var_data in all_variables.items():
        min_dates.append(var_data['date'].min())
        max_dates.append(var_data['date'].max())
    
    common_start = max(min_dates)
    common_end = min(max_dates)
    
    print(f"  Common date range: {common_start} to {common_end}")
    
    # Create a merged dataframe with all variables aligned by date
    aligned_data = pd.DataFrame({'date': pd.date_range(start=common_start, end=common_end)})
    
    variable_names = ['date']
    for var_name, var_data in all_variables.items():
        # Create a temporary dataframe with date and the variable
        temp_df = pd.DataFrame({
            'date': var_data['date'],
            var_name: var_data['data']
        })
        
        # Merge with aligned data
        aligned_data = pd.merge(aligned_data, temp_df, on='date', how='left')
        variable_names.append(var_name)
    
    # Handle missing values
    if aligned_data.isnull().values.any():
        print("  Warning: Data contains missing values. Filling with forward/backward fill.")
        # First try forward fill
        aligned_data = aligned_data.ffill()
        # Then backward fill for any remaining NaN
        aligned_data = aligned_data.bfill()
    
    # Get target index (excluding date column)
    target_idx = variable_names.index(target_name) - 1
    
    # Prepare dates and data arrays
    dates = aligned_data['date'].values
    features_data = aligned_data.drop(columns=['date']).values
    
    print(f"  Prepared dataset with {len(variable_names)-1} variables and {len(dates)} time points")
    print(f"  Target '{target_name}' is at index {target_idx}")
    
    return dates, features_data, target_idx, variable_names[1:]


# Add this function to your machine_learning_data.py file

def prepare_market_data(temporal_data, target_variable='Open'):
    """
    Prepare financial market data for prediction by finding the target variable
    and aligning all other relevant features.
    
    Args:
        temporal_data: Dictionary of dataframes from load_and_plot_financial_data
        target_variable: Variable to predict (e.g., 'Open', 'Close')
        
    Returns:
        dates: Array of dates
        data: Numpy array containing aligned data
        target_idx: Index of target variable in the data array
        variable_names: List of variable names
    """
    import numpy as np
    import pandas as pd
    
    # Find which dataframe contains the target variable
    target_df_key = None
    for key, df in temporal_data.items():
        if target_variable in df.columns:
            target_df_key = key
            break
    
    if target_df_key is None:
        raise ValueError(f"Target variable '{target_variable}' not found in any dataframe")
    
    # Get the target dataframe
    target_df = temporal_data[target_df_key]
    
    # Get dates from the target dataframe
    dates = target_df.index
    
    # Initialize lists for data and variable names
    data_columns = []
    variable_names = []
    
    # First add the target variable
    data_columns.append(target_df[target_variable].values)
    variable_names.append(f"{target_df_key}_{target_variable}")
    target_idx = 0  # Target is the first column
    
    # Then add all other variables
    for df_key, df in temporal_data.items():
        for col in df.columns:
            # Skip the target variable (already added)
            if df_key == target_df_key and col == target_variable:
                continue
            
            # Align the data with target dates
            aligned_series = df[col].reindex(dates)
            
            # Add the aligned data and variable name
            data_columns.append(aligned_series.values)
            variable_names.append(f"{df_key}_{col}")
    
    # Convert list of columns to numpy array
    data = np.column_stack(data_columns)
    
    # Handle missing values - use simple forward fill
    for j in range(data.shape[1]):
        col = data[:, j]
        mask = np.isnan(col)
        
        if mask.any():
            # Get indices of missing values
            indices = np.where(mask)[0]
            
            # For each index with missing value
            for idx in indices:
                # Find the nearest non-NaN value before this index
                prev_idx = idx - 1
                while prev_idx >= 0 and np.isnan(data[prev_idx, j]):
                    prev_idx -= 1
                
                if prev_idx >= 0:
                    # Forward fill
                    data[idx, j] = data[prev_idx, j]
                else:
                    # If no previous value, use the nearest future value
                    next_idx = idx + 1
                    while next_idx < len(data) and np.isnan(data[next_idx, j]):
                        next_idx += 1
                    
                    if next_idx < len(data):
                        data[idx, j] = data[next_idx, j]
                    else:
                        # If still no value found, use 0 (or other imputation)
                        data[idx, j] = 0
    
    return dates, data, target_idx, variable_names