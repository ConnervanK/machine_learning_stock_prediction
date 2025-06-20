'''
I need to create a tensor for a neural network.
the tensor needs to be created from data from different csv files where in the first column there is the date at which 
the data was collected in the format yy-mm-dd and in the second column the data itself. the first header is 
'date' and the second will have the name of the data.
I need a function that takes all the csv files and creates a dataframe with the first column the date column and then
the data columns with their respective names based on the headers. Use pandas, numpy, and torch to create the tensor.

'''


import pandas as pd
import numpy as np
import torch


def extend_monthly_data(df):
    """
    Extends monthly data to fill in daily values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with date column and potentially monthly data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with missing values filled in by month
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # First, check if we have NaN values that need filling
    if not result_df.iloc[:, 1:].isna().any().any():
        print("No missing values found, skipping monthly extension")
        return result_df
    
    # Check each column (except date)
    for col in result_df.columns:
        if col == 'date':
            continue
            
        # Check if this column has values primarily on the first day of the month
        values = result_df[~result_df[col].isna()]
        
        # If more than 70% of non-NaN values are on the 1st day, consider it monthly data
        first_day_values = values[values['date'].dt.day == 1]
        if len(first_day_values) >= 0.7 * len(values) and len(values) > 0:
            print(f"Column '{col}' appears to contain monthly data. Extending values...")
            
            # Group by year and month
            result_df['year_month'] = result_df['date'].dt.to_period('M')
            
            # For each month, fill forward the values
            for period in result_df['year_month'].unique():
                # Find the value for this month (if any)
                month_mask = result_df['year_month'] == period
                month_value = result_df.loc[month_mask, col].dropna().iloc[0] if any(~result_df.loc[month_mask, col].isna()) else None
                
                # If we have a value, fill it in for all days in this month
                if month_value is not None:
                    result_df.loc[month_mask, col] = month_value
            
            # Remove the temporary column
            result_df.drop(columns=['year_month'], inplace=True)
    
    print(f"After extending monthly data: {result_df.shape}")
    return result_df



def create_tensor_from_csvs(csv_files):
    """
    Create a tensor from multiple CSV files.
    
    Parameters:
    -----------
    csv_files : list
        List of paths to CSV files.
        
    Returns:
    --------
    torch.Tensor
        A tensor created from the combined data.
    pandas.DataFrame
        The combined dataframe with aligned dates.
    """
    # Step 1: Initialize an empty list to store individual dataframes
    dfs = []
    
    # Step 2: Read each CSV file and store in the list
    for file in csv_files:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Ensure the date column is properly named
        if 'date' not in df.columns:
            raise ValueError(f"CSV file {file} does not have a 'date' column")
        
        # Add the dataframe to our list
        dfs.append(df)
        
        # Print some information about this file
        print(f"Loaded {file} with columns: {df.columns.tolist()}")
    
    # Step 3: Make sure all date columns are in datetime format
    for i, df in enumerate(dfs):
        # Convert date column to datetime for proper merging
        dfs[i]['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        
    # Step 4: Merge all dataframes on date column
    # Start with the first dataframe
    if not dfs:
        raise ValueError("No valid CSV files were provided")
        
    merged_df = dfs[0]

    # Merge with the rest of dataframes one by one
    for i in range(1, len(dfs)):
        # Get all columns (excluding 'date')
        data_cols = [col for col in dfs[i].columns if col != 'date']
        
        # Merge on date column, including all data columns
        merged_df = pd.merge(
            merged_df, 
            dfs[i][['date'] + data_cols],  # Include date and all data columns
            on='date', 
            how='outer'  # Use outer join to keep all dates
        )

    # Step 5: Sort by date for consistency
    merged_df = merged_df.sort_values('date').reset_index(drop=True)

    # Step 6: Handle missing values by extending monthly data
    merged_df_clean = extend_monthly_data(merged_df)
    
    # Step 7: Handle any remaining NaN values
    # Option 1: Drop rows with any NaN values
    # merged_df_clean = merged_df_clean.dropna()
    
    # Option 2: Fill remaining NaNs with interpolation where possible
    for col in merged_df_clean.columns:
        if col != 'date':
            # Try to interpolate values (linear interpolation between known data points)
            merged_df_clean[col] = merged_df_clean[col].interpolate(method='linear')
    
    # Option 3: Forward/backward fill remaining NaNs at the edges
    # merged_df_clean = merged_df_clean.fillna(method='ffill').fillna(method='bfill')
    merged_df_clean = merged_df_clean.ffill().bfill()

    print(f"After handling NaN values: {merged_df_clean.shape}")

    # Step 8: Convert numeric data to tensor
    # Extract all columns except 'date'
    numeric_cols = [col for col in merged_df_clean.columns if col != 'date']
    numeric_data = merged_df_clean[numeric_cols].values.astype(np.float32)

    # Create tensor
    tensor_data = torch.tensor(numeric_data)
    print(f"Created tensor with shape: {tensor_data.shape}")

    print(merged_df_clean.head())  # Display the first few rows of the cleaned dataframe

    return tensor_data, merged_df_clean


# path = '../data/'  # Adjust this path as needed
# create_tensor_from_csvs(['stock_AAPL.csv', 'econ_GDP.csv'])


