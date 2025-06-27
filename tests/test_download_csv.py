import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from download_csv import download_financial_data, download_gdp_data


class TestDownloadFinancialData:
    
    @pytest.fixture
    def mock_download(self):
        with patch('download_csv.yf.download') as mock:
            yield mock
    def test_download_financial_data_basic(self, mock_download):
        """Test basic financial data download functionality"""
        # Mock yfinance download response
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'Close': [101, 102, 103],
            'High': [102, 103, 104],
            'Low': [99, 100, 101],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        mock_download.return_value = mock_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = 'test_financial.csv'
            
            # Test the function
            download_financial_data(['AAPL'], '2023-01-01', '2023-01-03', filename, tmpdir)
            
            # Check that file was created
            output_path = os.path.join(tmpdir, filename)
            assert os.path.exists(output_path)
            
            # Check file contents
            result_df = pd.read_csv(output_path)
            assert 'date' in result_df.columns
            assert 'Open' in result_df.columns
            assert len(result_df) == 3
    
    @patch('download_csv.yf.download')
    def test_download_financial_data_multi_ticker(self, mock_download):
        """Test download with multiple tickers (MultiIndex columns)"""
        # Create mock data with MultiIndex columns
        columns = pd.MultiIndex.from_tuples([
            ('AAPL', 'Open'), ('AAPL', 'Close'), ('AAPL', 'Volume')
        ])
        mock_data = pd.DataFrame(
            [[100, 101, 1000], [101, 102, 1100], [102, 103, 1200]],
            columns=columns,
            index=pd.date_range('2023-01-01', periods=3)
        )
        
        mock_download.return_value = mock_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = 'test_multi_ticker.csv'
            
            download_financial_data(['AAPL'], '2023-01-01', '2023-01-03', filename, tmpdir)
            
            # Check that file was created and columns were flattened
            output_path = os.path.join(tmpdir, filename)
            assert os.path.exists(output_path)
            
            result_df = pd.read_csv(output_path)
            assert 'date' in result_df.columns


class TestDownloadGDPData:
    
    @patch('download_csv.pdr.DataReader')
    def test_download_gdp_data_basic(self, mock_datareader):
        """Test basic GDP data download functionality"""
        # Mock pandas_datareader response
        mock_data = pd.DataFrame({
            'GDP': [20000, 20100, 20200]
        }, index=pd.to_datetime(['2023-01-01', '2023-04-01', '2023-07-01']))
        mock_data.index.name = 'DATE'
        
        mock_datareader.return_value = mock_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = 'test_gdp.csv'
            
            # Test the function
            download_gdp_data('2023-01-01', '2023-12-31', filename, tmpdir)
            
            # Check that file was created
            output_path = os.path.join(tmpdir, filename)
            assert os.path.exists(output_path)
            
            # Check file contents
            result_df = pd.read_csv(output_path)
            assert 'date' in result_df.columns
            assert 'gdp' in result_df.columns
    
    @patch('download_csv.pdr.DataReader')
    def test_download_gdp_data_error_handling(self, mock_datareader):
        """Test error handling in GDP data download"""
        mock_datareader.side_effect = Exception("API Error")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = 'test_gdp_error.csv'
            
            # Should handle the exception gracefully (not crash)
            with pytest.raises(Exception):
                download_gdp_data('2023-01-01', '2023-12-31', filename, tmpdir)


class TestDownloadIntegration:
    
    def test_folder_creation(self):
        """Test that the download functions create output folders if they don't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a non-existent subfolder path
            output_folder = os.path.join(tmpdir, 'new_folder', 'subfolder')
            
            with patch('download_csv.yf.download') as mock_download:
                mock_data = pd.DataFrame({
                    'Open': [100], 'Close': [101]
                }, index=pd.date_range('2023-01-01', periods=1))
                mock_download.return_value = mock_data
                
                download_financial_data(['AAPL'], '2023-01-01', '2023-01-01', 'test.csv', output_folder)
                
                # Check that the folder was created
                assert os.path.exists(output_folder)
                assert os.path.exists(os.path.join(output_folder, 'test.csv'))


if __name__ == "__main__":
    pytest.main([__file__])
