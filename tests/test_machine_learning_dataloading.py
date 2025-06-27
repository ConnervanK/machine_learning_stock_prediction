import pytest
import pandas as pd
import numpy as np
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from machine_learning_dataloading import extend_monthly_data, create_tensor_from_csvs


class TestExtendMonthlyDataLoading:
    
    def test_extend_monthly_data_basic(self):
        """Test basic monthly data extension functionality"""
        # Create sample data with monthly pattern
        dates = pd.date_range('2023-01-01', periods=31, freq='D')
        data = [100.0] + [np.nan] * 30  # Only first day has value
        
        df = pd.DataFrame({
            'date': dates,
            'gdp': data
        })
        
        result = extend_monthly_data(df)
        
        # Check that function processed the data
        assert isinstance(result, pd.DataFrame)
        assert 'date' in result.columns
        assert 'gdp' in result.columns
        assert len(result) == len(df)
    
    def test_extend_monthly_data_no_missing(self):
        """Test extend_monthly_data with complete data"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'price': range(100, 110)
        })
        
        result = extend_monthly_data(df)
        
        # Should return similar dataframe
        assert len(result) == len(df)
        assert list(result.columns) == list(df.columns)


class TestCreateTensorFromCSVsDataLoading:
    
    def test_create_tensor_from_csvs_simple(self):
        """Test basic tensor creation from CSV files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV files
            csv1_path = os.path.join(tmpdir, 'test1.csv')
            csv2_path = os.path.join(tmpdir, 'test2.csv')
            
            # Create sample data
            dates = pd.date_range('2023-01-01', periods=5, freq='D')
            
            df1 = pd.DataFrame({
                'date': dates.strftime('%Y-%m-%d'),
                'value1': [1, 2, 3, 4, 5]
            })
            
            df2 = pd.DataFrame({
                'date': dates.strftime('%Y-%m-%d'),
                'value2': [10, 20, 30, 40, 50]
            })
            
            df1.to_csv(csv1_path, index=False)
            df2.to_csv(csv2_path, index=False)
            
            # Test the function
            tensor, merged_df = create_tensor_from_csvs([csv1_path, csv2_path])
            
            # Validate results
            assert isinstance(tensor, torch.Tensor)
            assert isinstance(merged_df, pd.DataFrame)
            assert tensor.shape[0] == 5  # 5 rows
            assert tensor.shape[1] == 2  # 2 data columns
            assert 'date' in merged_df.columns
            assert 'value1' in merged_df.columns
            assert 'value2' in merged_df.columns
    
    def test_create_tensor_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with empty file list
        with pytest.raises(ValueError):
            create_tensor_from_csvs([])
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            create_tensor_from_csvs(['non_existent_file.csv'])
    
    def test_create_tensor_missing_date_column(self):
        """Test handling of CSV without date column"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'no_date.csv')
            
            df = pd.DataFrame({
                'value': [1, 2, 3]
            })
            df.to_csv(csv_path, index=False)
            
            with pytest.raises(ValueError, match="does not have a 'date' column"):
                create_tensor_from_csvs([csv_path])
    
    def test_create_tensor_with_nans(self):
        """Test tensor creation with missing values"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'with_nans.csv')
            
            dates = pd.date_range('2023-01-01', periods=5, freq='D')
            df = pd.DataFrame({
                'date': dates.strftime('%Y-%m-%d'),
                'value': [1.0, np.nan, 3.0, np.nan, 5.0]
            })
            df.to_csv(csv_path, index=False)
            
            tensor, merged_df = create_tensor_from_csvs([csv_path])
            
            # Check that NaNs were handled
            assert not torch.isnan(tensor).any()
            assert not merged_df['value'].isna().any()


class TestDataLoadingPipeline:
    
    def test_full_pipeline_simulation(self):
        """Test a complete data loading pipeline simulation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple CSV files simulating real data
            files = []
            
            # Financial data
            financial_path = os.path.join(tmpdir, 'financial.csv')
            dates = pd.date_range('2023-01-01', periods=10, freq='D')
            financial_df = pd.DataFrame({
                'date': dates.strftime('%Y-%m-%d'),
                'Open': np.random.uniform(100, 200, 10),
                'Close': np.random.uniform(100, 200, 10)
            })
            financial_df.to_csv(financial_path, index=False)
            files.append(financial_path)
            
            # Economic data
            economic_path = os.path.join(tmpdir, 'economic.csv')
            economic_df = pd.DataFrame({
                'date': dates.strftime('%Y-%m-%d'),
                'gdp': np.random.uniform(20000, 21000, 10)
            })
            economic_df.to_csv(economic_path, index=False)
            files.append(economic_path)
            
            # Test the pipeline
            tensor, merged_df = create_tensor_from_csvs(files)
            
            # Validate the results
            assert tensor.shape[0] == 10  # 10 time steps
            assert tensor.shape[1] == 3   # Open, Close, gdp
            assert len(merged_df) == 10
            assert 'date' in merged_df.columns
            assert 'Open' in merged_df.columns
            assert 'Close' in merged_df.columns
            assert 'gdp' in merged_df.columns


if __name__ == "__main__":
    pytest.main([__file__])
