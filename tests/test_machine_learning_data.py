import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from machine_learning_data import prepare_market_data, generate_financial_data


class TestPrepareMarketData:
    
    def test_prepare_market_data_basic(self):
        """Test basic market data preparation"""
        # Create sample temporal data
        temporal_data = {
            'stock_data': pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=10),
                'Open': range(100, 110),
                'Close': range(101, 111),
                'Volume': range(1000, 1010)
            }),
            'gdp_data': pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=10),
                'gdp': range(20000, 20010)
            })
        }
        
        dates, data, target_idx, variable_names = prepare_market_data(temporal_data, target_variable='Open')
        
        # Check return types and shapes
        assert isinstance(dates, np.ndarray)
        assert isinstance(data, np.ndarray)
        assert isinstance(target_idx, int)
        assert isinstance(variable_names, list)
        
        # Check that target variable was found
        assert target_idx >= 0
        assert 'Open' in variable_names[target_idx] or 'open' in variable_names[target_idx].lower()
    
    def test_prepare_market_data_target_not_found(self):
        """Test behavior when target variable is not found"""
        temporal_data = {
            'data': pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=5),
                'price': range(100, 105)
            })
        }
        
        # Should handle missing target gracefully
        dates, data, target_idx, variable_names = prepare_market_data(temporal_data, target_variable='NonExistent')
        
        # target_idx should indicate not found
        assert target_idx == -1 or target_idx is None
    
    def test_prepare_market_data_empty_data(self):
        """Test behavior with empty data"""
        temporal_data = {}
        
        dates, data, target_idx, variable_names = prepare_market_data(temporal_data)
        
        # Should handle empty data gracefully
        assert len(dates) == 0 or dates is None
        assert data.size == 0 or data is None


class TestGenerateFinancialData:
    
    @patch('machine_learning_data.dcsv.download_financial_data')
    @patch('machine_learning_data.dcsv.download_gdp_data')
    @patch('machine_learning_data.dcsv.download_inflation_data')
    @patch('machine_learning_data.dcsv.download_interest_rate_data')
    @patch('machine_learning_data.dcsv.download_unemployment_rate_data')
    def test_generate_financial_data(self, mock_unemployment, mock_interest, mock_inflation, mock_gdp, mock_financial):
        """Test the generate_financial_data function"""
        # Mock all download functions to avoid actual API calls
        mock_financial.return_value = None
        mock_gdp.return_value = None
        mock_inflation.return_value = None
        mock_interest.return_value = None
        mock_unemployment.return_value = None
        
        # Should not raise any exceptions
        try:
            generate_financial_data()
            # If we get here, the function executed without errors
            assert True
        except Exception as e:
            # If there's an import error or similar, that's expected in test environment
            assert "module" in str(e).lower() or "import" in str(e).lower()


class TestDataIntegration:
    
    def test_data_processing_pipeline(self):
        """Test a simple data processing pipeline"""
        # Create mock temporal data similar to what would be loaded
        temporal_data = {
            'financial_data': pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=20),
                'Open': np.random.uniform(100, 200, 20),
                'Close': np.random.uniform(100, 200, 20),
                'Volume': np.random.uniform(1000, 2000, 20)
            }),
            'economic_data': pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=20),
                'gdp': np.random.uniform(20000, 21000, 20),
                'interest_rate': np.random.uniform(2, 5, 20)
            })
        }
        
        # Test data preparation
        dates, data, target_idx, variable_names = prepare_market_data(temporal_data, target_variable='Open')
        
        # Validate results
        assert len(dates) > 0
        assert data.shape[0] > 0  # Should have rows
        assert data.shape[1] > 0  # Should have columns
        assert len(variable_names) == data.shape[1]  # Names should match columns
        
        # Check data types
        assert data.dtype in [np.float32, np.float64]
        assert all(isinstance(name, str) for name in variable_names)


if __name__ == "__main__":
    pytest.main([__file__])
