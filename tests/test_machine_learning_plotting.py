import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from machine_learning_plotting import (
    load_and_plot_temporal_data, 
    plot_temporal_data,
    create_half_correlation_plot3,
    plot_financial_data_from_tensor
)


class TestLoadAndPlotTemporalData:
    
    def test_load_and_plot_temporal_data_basic(self):
        """Test basic loading of temporal data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test CSV file
            csv_path = os.path.join(tmpdir, 'test_data.csv')
            
            dates = pd.date_range('2023-01-01', periods=10, freq='D')
            df = pd.DataFrame({
                'date': dates,
                'price': range(100, 110),
                'volume': range(1000, 1010)
            })
            df.to_csv(csv_path, index=False)
            
            # Test the function (suppress plotting)
            with patch('matplotlib.pyplot.show'):
                result = load_and_plot_temporal_data(tmpdir)
            
            # Check results
            assert isinstance(result, dict)
            assert 'test_data.csv' in result
            assert 'df' in result['test_data.csv']
            assert isinstance(result['test_data.csv']['df'], pd.DataFrame)
    
    def test_load_and_plot_temporal_data_empty_folder(self):
        """Test behavior with empty folder"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_and_plot_temporal_data(tmpdir)
            
            # Should return empty dict
            assert isinstance(result, dict)
            assert len(result) == 0
    
    def test_load_and_plot_temporal_data_nonexistent_folder(self):
        """Test behavior with non-existent folder"""
        result = load_and_plot_temporal_data('/non/existent/path')
        
        # Should return empty dict
        assert isinstance(result, dict)
        assert len(result) == 0


class TestPlotTemporalData:
    
    def test_plot_temporal_data_with_date_column(self):
        """Test plotting temporal data with date column"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'price': range(100, 110),
            'volume': range(1000, 1010)
        })
        
        # Test without showing plots
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_temporal_data(df, date_col='date', filename='test')
            
            # Should have called show multiple times (one per numeric column + combined plot)
            assert mock_show.call_count >= 2
    
    def test_plot_temporal_data_without_date_column(self):
        """Test plotting temporal data without date column"""
        df = pd.DataFrame({
            'price': range(100, 110),
            'volume': range(1000, 1010)
        })
        
        # Test without showing plots
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_temporal_data(df, date_col=None, filename='test')
            
            # Should still plot successfully
            assert mock_show.call_count >= 2


class TestCreateHalfCorrelationPlot3:
    
    def test_create_half_correlation_plot3_basic(self):
        """Test basic correlation plot creation"""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        loaded_data = pd.DataFrame({
            'date': dates,
            'Close': np.random.uniform(100, 200, 50),
            'Volume': np.random.uniform(1000, 2000, 50),
            'gdp': np.random.uniform(20000, 21000, 50),
            'interest_rate': np.random.uniform(2, 5, 50)
        })
        
        # Test without showing/saving plots
        with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.savefig'):
            corr_matrix, found_vars, focused_df = create_half_correlation_plot3(
                loaded_data, plot=False, save=False
            )
        
        # Check results
        assert isinstance(corr_matrix, pd.DataFrame)
        assert isinstance(found_vars, list)
        assert isinstance(focused_df, pd.DataFrame)
        assert len(found_vars) > 0
        assert corr_matrix.shape[0] == corr_matrix.shape[1]  # Square matrix
    
    def test_create_half_correlation_plot3_dict_input(self):
        """Test correlation plot with dictionary input"""
        # Create sample data as dictionary (multiple datasets)
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        loaded_data = {
            'stock_data': pd.DataFrame({
                'Close': np.random.uniform(100, 200, 20),
                'Volume': np.random.uniform(1000, 2000, 20)
            }),
            'economic_data': pd.DataFrame({
                'gdp': np.random.uniform(20000, 21000, 20)
            })
        }
        
        # Test without showing/saving plots
        with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.savefig'):
            corr_matrix, found_vars, focused_df = create_half_correlation_plot3(
                loaded_data, plot=False, save=False
            )
        
        # Check results
        assert isinstance(corr_matrix, pd.DataFrame)
        assert isinstance(found_vars, list)
        assert len(found_vars) > 0


class TestPlotFinancialDataFromTensor:
    
    def test_plot_financial_data_from_tensor_basic(self):
        """Test plotting financial data from tensor format"""
        # Create sample merged dataframe (output from create_tensor_from_csvs)
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        merged_df = pd.DataFrame({
            'date': dates,
            'Open': np.random.uniform(100, 200, 20),
            'Close': np.random.uniform(100, 200, 20),
            'Volume': np.random.uniform(1000, 2000, 20),
            'gdp': np.random.uniform(20000, 21000, 20),
            'interest_rate': np.random.uniform(2, 5, 20)
        })
        
        # Test without showing plots
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_financial_data_from_tensor(merged_df, plot=True)
            
            # Should have called show (plots were generated)
            assert mock_show.call_count > 0
    
    def test_plot_financial_data_from_tensor_no_plot(self):
        """Test data processing without plotting"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        merged_df = pd.DataFrame({
            'date': dates,
            'price': np.random.uniform(100, 200, 10)
        })
        
        # Test without plotting
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_financial_data_from_tensor(merged_df, plot=False)
            
            # Should not have called show
            assert mock_show.call_count == 0


class TestPlottingIntegration:
    
    def test_plotting_pipeline(self):
        """Test a complete plotting pipeline"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data files
            csv_path = os.path.join(tmpdir, 'financial_data.csv')
            dates = pd.date_range('2023-01-01', periods=30, freq='D')
            df = pd.DataFrame({
                'date': dates,
                'Open': np.random.uniform(100, 200, 30),
                'Close': np.random.uniform(100, 200, 30),
                'Volume': np.random.uniform(1000, 2000, 30)
            })
            df.to_csv(csv_path, index=False)
            
            # Test full pipeline
            with patch('matplotlib.pyplot.show'):
                # Load data
                loaded_data = load_and_plot_temporal_data(tmpdir)
                
                # Should have loaded data
                assert len(loaded_data) > 0
                
                # Extract dataframe for tensor plotting
                first_key = list(loaded_data.keys())[0]
                test_df = loaded_data[first_key]['df']
                
                # Add date column for tensor plotting format
                test_df['date'] = dates
                
                # Test tensor plotting
                plot_financial_data_from_tensor(test_df, plot=False)
                
                # If we get here, all functions executed successfully
                assert True


if __name__ == "__main__":
    pytest.main([__file__])
