import pytest
import pandas as pd
import numpy as np
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from create_tensor import extend_monthly_data, create_tensor_from_csvs


class TestExtendMonthlyData:
    
    def test_extend_monthly_data_basic(self):
        """Test basic functionality of extend_monthly_data"""
        # Create sample data with monthly values
        dates = pd.date_range('2023-01-01', periods=31, freq='D')
        data = [100.0] + [np.nan] * 30  # Only first day has value
        
        df = pd.DataFrame({
            'date': dates,
            'gdp': data
        })
        
        result = extend_monthly_data(df)
        
        # Check that NaN values were filled
        assert not result['gdp'].isna().any()
        assert result['gdp'].iloc[0] == 100.0
        assert result['gdp'].iloc[15] == 100.0  # Mid-month should be filled
    
    def test_extend_monthly_data_no_missing_values(self):
        """Test extend_monthly_data with no missing values"""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = [100.0, 101.0, 102.0, 103.0, 104.0]
        
        df = pd.DataFrame({
            'date': dates,
            'price': data
        })
        
        result = extend_monthly_data(df)
        
        # Should return unchanged data
        pd.testing.assert_frame_equal(result, df)


class TestCreateTensorFromCSVs:
    
    def test_create_tensor_from_csvs_basic(self):
        """Test basic tensor creation from CSV files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV files
            csv1_path = os.path.join(tmpdir, 'data1.csv')
            csv2_path = os.path.join(tmpdir, 'data2.csv')
            
            # Create sample data
            dates = pd.date_range('2023-01-01', periods=10, freq='D')
            
            df1 = pd.DataFrame({
                'date': dates,
                'price': range(10)
            })
            
            df2 = pd.DataFrame({
                'date': dates,
                'volume': range(100, 110)
            })
            
            df1.to_csv(csv1_path, index=False)
            df2.to_csv(csv2_path, index=False)
            
            # Test the function
            tensor, merged_df = create_tensor_from_csvs([csv1_path, csv2_path])
            
            # Check tensor properties
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32
            assert tensor.shape[0] == 10  # 10 days
            assert tensor.shape[1] == 2   # 2 columns (price, volume)
            
            # Check dataframe properties
            assert isinstance(merged_df, pd.DataFrame)
            assert 'date' in merged_df.columns
            assert 'price' in merged_df.columns
            assert 'volume' in merged_df.columns
            assert len(merged_df) == 10
    
    def test_create_tensor_from_csvs_missing_date_column(self):
        """Test error handling when date column is missing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'bad_data.csv')
            
            # Create CSV without date column
            df = pd.DataFrame({
                'price': [1, 2, 3],
                'volume': [10, 20, 30]
            })
            df.to_csv(csv_path, index=False)
            
            # Should raise ValueError
            with pytest.raises(ValueError, match="does not have a 'date' column"):
                create_tensor_from_csvs([csv_path])
    
    def test_create_tensor_from_csvs_empty_file_list(self):
        """Test error handling with empty file list"""
        with pytest.raises(ValueError, match="No valid CSV files were provided"):
            create_tensor_from_csvs([])
    
    def test_create_tensor_from_csvs_with_nans(self):
        """Test tensor creation with NaN values"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'data_with_nans.csv')
            
            dates = pd.date_range('2023-01-01', periods=5, freq='D')
            df = pd.DataFrame({
                'date': dates,
                'price': [1.0, np.nan, 3.0, np.nan, 5.0]
            })
            df.to_csv(csv_path, index=False)
            
            tensor, merged_df = create_tensor_from_csvs([csv_path])
            
            # Check that NaNs were handled (should be interpolated/filled)
            assert not torch.isnan(tensor).any()
            assert not merged_df['price'].isna().any()


if __name__ == "__main__":
    pytest.main([__file__])
