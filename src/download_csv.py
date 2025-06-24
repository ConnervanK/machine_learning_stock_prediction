'''
I need to create different functions that downloads finance data to a csv file. where in the first column is the date
in yyyy-mm-dd format named 'date' and the second will have the name or names of the data.
The sources are the following:
gdp, inflation, interest rate, unemployment rate, and exchange rate.
Furthermore, I want the sentiment data to be downloaded in a similar way
'''
import pandas as pd
from pandas_datareader import data as pdr

import yfinance as yf
from datetime import datetime, timedelta
import os
from typing import List, Dict

def download_financial_data(tickers: List[str], start_date: str, end_date: str, filename: str, folder: str) -> None:
    """
    Downloads financial data for given tickers and saves it to a CSV file.
    
    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols to download data for.
    start_date : str
        Start date in 'yyyy-mm-dd' format.
    end_date : str
        End date in 'yyyy-mm-dd' format.
    filename : str
        Name of the output CSV file.
    folder : str
        Folder path where the file will be saved.
    """
    # Download data using yfinance
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    
    # Flatten multi-level columns if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]

    # Remove ticker prefix from column names (e.g., "^GSPC_")
    ticker_prefix = tickers[0] + "_"
    data.columns = [col.replace(ticker_prefix, '') for col in data.columns]

    # Reset index to have 'date' as a column
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'date'}, inplace=True)

    # Ensure output directory exists
    os.makedirs(folder, exist_ok=True)

    # Save to CSV without extra header row
    output_path = os.path.join(folder, filename)
    data.to_csv(output_path, index=False)

    # print(data.head())


# # Define the tickers and date range
# # tickers = ['^GSPC', 'EURUSD=X', 'BTC-USD']  # Example tickers: S&P 500, EUR/USD exchange rate, Bitcoin
# tickers = ['^GSPC']  # Example tickers: S&P 500, EUR/USD exchange rate, Bitcoin
# start_date = '2020-01-01'
# end_date = datetime.now().strftime('%Y-%m-%d')  # Current date

# # Define the folder to save the CSV files
# folder = 'data'  # Ensure this folder exists or create it
# if not os.path.exists(folder):
#     os.makedirs(folder)
# # Download financial data
# download_financial_data(tickers, start_date, end_date, 'financial_data.csv', folder)

def download_gdp_data(start_date: str, end_date: str, filename: str, folder: str) -> None:
    """
    Downloads GDP data from FRED and saves it to a CSV file.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'yyyy-mm-dd' format.
    end_date : str
        End date in 'yyyy-mm-dd' format.
    filename : str
        Name of the output CSV file.
    folder : str
        Folder path where the file will be saved.
    """
    # GDP data from FRED (quarterly, code: 'GDP')
    gdp_data = pdr.DataReader('GDP', 'fred', start_date, end_date)
    
    # Reset index to have 'date' as a column
    gdp_data.reset_index(inplace=True)
    gdp_data.rename(columns={'DATE': 'date', 'GDP': 'gdp'}, inplace=True)

    # Set 'date' as datetime index for asfreq to work
    gdp_data['date'] = pd.to_datetime(gdp_data['date'])
    gdp_data.set_index('date', inplace=True)

    # if gdp data is not daily, we can interpolate it to daily frequency
    gdp_data = gdp_data.asfreq('D', method='ffill')

    # Reset index to have 'date' as a column again
    gdp_data.reset_index(inplace=True)
    
    # Ensure output directory exists
    os.makedirs(folder, exist_ok=True)

    # Save to CSV without extra header row
    output_path = os.path.join(folder, filename)
    gdp_data.to_csv(output_path, index=False)

# # example usage
# start_date = '2020-01-01'
# end_date = datetime.now().strftime('%Y-%m-%d')
# download_gdp_data(start_date, end_date, 'gdp_data.csv', folder='data')

def download_inflation_data(start_date: str, end_date: str, filename: str, folder: str) -> None:
    """
    Downloads inflation data from FRED and saves it to a CSV file.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'yyyy-mm-dd' format.
    end_date : str
        End date in 'yyyy-mm-dd' format.
    filename : str
        Name of the output CSV file.
    folder : str
        Folder path where the file will be saved.
    """
    # Inflation data from FRED (CPI, code: 'CPIAUCNS')
    inflation_data = pdr.DataReader('CPIAUCNS', 'fred', start_date, end_date)
    
    # Reset index to have 'date' as a column
    inflation_data.reset_index(inplace=True)
    inflation_data.rename(columns={'DATE': 'date', 'CPIAUCNS': 'inflation'}, inplace=True)

    # Set 'date' as datetime index for asfreq to work
    inflation_data['date'] = pd.to_datetime(inflation_data['date'])
    inflation_data.set_index('date', inplace=True)

    # if inflation data is not daily, we can interpolate it to daily frequency
    inflation_data = inflation_data.asfreq('D', method='ffill')

    # Reset index to have 'date' as a column again
    inflation_data.reset_index(inplace=True)
    
    # Ensure output directory exists
    os.makedirs(folder, exist_ok=True)

    # Save to CSV without extra header row
    output_path = os.path.join(folder, filename)
    inflation_data.to_csv(output_path, index=False)

# # example usage
# start_date = '2020-01-01'
# end_date = datetime.now().strftime('%Y-%m-%d')
# download_inflation_data(start_date, end_date, 'inflation_data.csv', folder='data')

def download_interest_rate_data(start_date: str, end_date: str, filename: str, folder: str) -> None:
    """
    Downloads interest rate data from FRED and saves it to a CSV file.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'yyyy-mm-dd' format.
    end_date : str
        End date in 'yyyy-mm-dd' format.
    filename : str
        Name of the output CSV file.
    folder : str
        Folder path where the file will be saved.
    """
    # Interest rate data from FRED (3-Month Treasury Bill, code: 'DTB3')
    interest_rate_data = pdr.DataReader('DTB3', 'fred', start_date, end_date)
    
    # Reset index to have 'date' as a column
    interest_rate_data.reset_index(inplace=True)
    interest_rate_data.rename(columns={'DATE': 'date', 'DTB3': 'interest_rate'}, inplace=True)

    # Set 'date' as datetime index for asfreq to work
    interest_rate_data['date'] = pd.to_datetime(interest_rate_data['date'])
    interest_rate_data.set_index('date', inplace=True)

    # Interpolate missing values after forward fill for weekends/holidays
    interest_rate_data = interest_rate_data.asfreq('D', method='ffill')
    interest_rate_data['interest_rate'] = interest_rate_data['interest_rate'].interpolate(method='linear')

    # Reset index to have 'date' as a column again
    interest_rate_data.reset_index(inplace=True)
    
    # Ensure output directory exists
    os.makedirs(folder, exist_ok=True)

    # Save to CSV without extra header row
    output_path = os.path.join(folder, filename)
    interest_rate_data.to_csv(output_path, index=False)

# # example usage
# start_date = '2020-01-01'
# end_date = datetime.now().strftime('%Y-%m-%d')
# download_interest_rate_data(start_date, end_date, 'interest_rate_data.csv', folder='data')

def download_unemployment_rate_data(start_date: str, end_date: str, filename: str, folder: str) -> None:
    """
    Downloads unemployment rate data from FRED and saves it to a CSV file.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'yyyy-mm-dd' format.
    end_date : str
        End date in 'yyyy-mm-dd' format.
    filename : str
        Name of the output CSV file.
    folder : str
        Folder path where the file will be saved.
    """
    # Unemployment rate data from FRED (code: 'UNRATE')
    unemployment_data = pdr.DataReader('UNRATE', 'fred', start_date, end_date)
    
    # Reset index to have 'date' as a column
    unemployment_data.reset_index(inplace=True)
    unemployment_data.rename(columns={'DATE': 'date', 'UNRATE': 'unemployment_rate'}, inplace=True)

    # Set 'date' as datetime index for asfreq to work
    unemployment_data['date'] = pd.to_datetime(unemployment_data['date'])
    unemployment_data.set_index('date', inplace=True)

    # if unemployment data is not daily, we can interpolate it to daily frequency
    unemployment_data = unemployment_data.asfreq('D', method='ffill')

    # Reset index to have 'date' as a column again
    unemployment_data.reset_index(inplace=True)
    
    # Ensure output directory exists
    os.makedirs(folder, exist_ok=True)

    # Save to CSV without extra header row
    output_path = os.path.join(folder, filename)
    unemployment_data.to_csv(output_path, index=False)
    
# # example usage
# start_date = '2020-01-01'
# end_date = datetime.now().strftime('%Y-%m-%d')
# download_unemployment_rate_data(start_date, end_date, 'unemployment_rate_data.csv', folder='data')



def download_exchange_rate_data(start_date: str, end_date: str, filename: str, folder: str) -> None:
    """
    Downloads exchange rate data from FRED and saves it to a CSV file.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'yyyy-mm-dd' format.
    end_date : str
        End date in 'yyyy-mm-dd' format.
    filename : str
        Name of the output CSV file.
    folder : str
        Folder path where the file will be saved.
    """
    # Exchange rate data from FRED (Euro to USD, code: 'DEXUSEU')
    exchange_rate_data = pdr.DataReader('DEXUSEU', 'fred', start_date, end_date)
    
    # Reset index to have 'date' as a column
    exchange_rate_data.reset_index(inplace=True)
    exchange_rate_data.rename(columns={'DATE': 'date', 'DEXUSEU': 'exchange_rate'}, inplace=True)

    # Set 'date' as datetime index for asfreq to work
    exchange_rate_data['date'] = pd.to_datetime(exchange_rate_data['date'])
    exchange_rate_data.set_index('date', inplace=True)

    # if exchange rate data is not daily, we can interpolate it to daily frequency
    exchange_rate_data = exchange_rate_data.asfreq('D')
    exchange_rate_data.ffill(inplace=True)

    # Reset index to have 'date' as a column again
    exchange_rate_data.reset_index(inplace=True)
    
    # Ensure output directory exists
    os.makedirs(folder, exist_ok=True)

    # Save to CSV without extra header row
    output_path = os.path.join(folder, filename)
    exchange_rate_data.to_csv(output_path, index=False)

# # example usage
# start_date = '2020-01-01'
# end_date = datetime.now().strftime('%Y-%m-%d')
# download_exchange_rate_data(start_date, end_date, 'exchange_rate_data.csv', folder='data')



