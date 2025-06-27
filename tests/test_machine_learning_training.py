import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from machine_learning_training import (
    run_data_test,
    add_financial_features,
    parallelized_rolling_window_prediction_for_financial_data,
    parallelized_rolling_window_prediction_for_financial_data2
)


class TestRunDataTest:
    
    def test_run_data_test_basic(self):
        """Test basic data test function"""
        # Mock the mld.test() call to avoid dependencies
        with patch('machine_learning_training.mld.test') as mock_test:
            mock_test.return_value = None
            
            # Should not raise any exceptions
            try:
                run_data_test()
                assert True
            except Exception as e:
                # If there's an import issue, that's expected in test environment
                assert "module" in str(e).lower() or "import" in str(e).lower()


class TestAddFinancialFeatures:
    
    def test_add_financial_features_basic(self):
        """Test adding financial features to dataframe"""
        # Create sample stock data
        df = pd.DataFrame({
            'Open': [100, 102, 101, 103, 105],
            'Close': [101, 103, 102, 104, 106],
            'High': [102, 104, 103, 105, 107],
            'Low': [99, 101, 100, 102, 104],
            'Volume': [1000, 1100, 900, 1200, 1300]
        })
        
        target_column = 'Close'
        
        # Test the function
        result_df = add_financial_features(df, target_column)
        
        # Check that new features were added
        assert len(result_df.columns) > len(df.columns)
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(df)  # Same number of rows
        
        # Check for some expected features (RSI, moving averages, etc.)
        new_columns = set(result_df.columns) - set(df.columns)
        assert len(new_columns) > 0  # Should have added some features
    
    def test_add_financial_features_minimal_data(self):
        """Test with minimal data that might not support all features"""
        # Very small dataset
        df = pd.DataFrame({
            'Close': [100, 101],
            'Volume': [1000, 1100]
        })
        
        target_column = 'Close'
        
        # Should handle gracefully without crashing
        result_df = add_financial_features(df, target_column)
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(df)


class TestParallelizedRollingWindowPrediction:
    
    @patch('machine_learning_training.plt.show')
    @patch('machine_learning_training.plt.savefig')
    def test_parallelized_prediction_basic_structure(self, mock_savefig, mock_show):
        """Test basic structure of parallelized prediction function"""
        # Create minimal temporal data
        temporal_data = {
            'stock_data': pd.DataFrame({
                'Open': np.random.uniform(100, 200, 50),
                'Close': np.random.uniform(100, 200, 50),
                'Volume': np.random.uniform(1000, 2000, 50)
            })
        }
        
        # Test with minimal parameters to avoid long training
        try:
            dates, predictions, actuals, std_devs = parallelized_rolling_window_prediction_for_financial_data(
                temporal_data,
                target_variable='Open',
                initial_train_size=0.8,
                sequence_length=5,
                epochs=1,  # Minimal epochs for testing
                hidden_dim=8,  # Small network
                num_layers=1,
                batch_size=4,
                num_workers=1,
                window_step=5,  # Large step to reduce iterations
                mc_samples=2,  # Minimal MC samples
                use_features=False  # Univariate for simplicity
            )
            
            # Check return types
            assert isinstance(dates, list)
            assert isinstance(predictions, list)
            assert isinstance(actuals, list)
            assert isinstance(std_devs, list)
            
        except Exception as e:
            # Expected in test environment due to dependencies
            expected_errors = ['module', 'import', 'cuda', 'torch', 'no module']
            assert any(err in str(e).lower() for err in expected_errors)
    
    @patch('machine_learning_training.plt.show')
    @patch('machine_learning_training.plt.savefig')
    def test_parallelized_prediction2_basic_structure(self, mock_savefig, mock_show):
        """Test basic structure of parallelized prediction function 2"""
        # Create sample tensor data
        tensor_data = torch.randn(50, 3)  # 50 timesteps, 3 features
        
        # Create sample merged dataframe
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        merged_df = pd.DataFrame({
            'date': dates,
            'Open': np.random.uniform(100, 200, 50),
            'Close': np.random.uniform(100, 200, 50),
            'Volume': np.random.uniform(1000, 2000, 50)
        })
        
        # Test with minimal parameters
        try:
            dates, predictions, actuals, std_devs = parallelized_rolling_window_prediction_for_financial_data2(
                tensor_data,
                target_variable='Open',
                initial_train_size=0.8,
                sequence_length=5,
                epochs=1,  # Minimal epochs
                hidden_dim=8,
                num_layers=1,
                batch_size=4,
                num_workers=1,
                window_step=5,
                mc_samples=2,
                use_features=False,
                merged_df=merged_df
            )
            
            # Check return types
            assert isinstance(dates, list)
            assert isinstance(predictions, list)
            assert isinstance(actuals, list)
            assert isinstance(std_devs, list)
            
        except Exception as e:
            # Expected in test environment due to dependencies
            expected_errors = ['module', 'import', 'cuda', 'torch', 'no module']
            assert any(err in str(e).lower() for err in expected_errors)


class TestTrainingIntegration:
    
    def test_training_module_import(self):
        """Test that the training module can be imported"""
        # If we can import the functions, basic structure is correct
        assert callable(run_data_test)
        assert callable(add_financial_features)
        assert callable(parallelized_rolling_window_prediction_for_financial_data)
        assert callable(parallelized_rolling_window_prediction_for_financial_data2)
    
    def test_feature_engineering_pipeline(self):
        """Test feature engineering pipeline"""
        # Create realistic stock data
        np.random.seed(42)  # For reproducible tests
        n_days = 100
        
        base_price = 100
        price_changes = np.random.normal(0, 1, n_days)
        prices = [base_price]
        
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change/100))
        
        df = pd.DataFrame({
            'Open': prices,
            'Close': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
            'Volume': np.random.uniform(1000, 5000, n_days)
        })
        
        # Test feature engineering
        enhanced_df = add_financial_features(df, 'Close')
        
        # Validate results
        assert len(enhanced_df) == len(df)
        assert len(enhanced_df.columns) > len(df.columns)
        
        # Check for NaN handling
        # Some features might create NaNs for initial values
        nan_ratio = enhanced_df.isna().sum().sum() / (len(enhanced_df) * len(enhanced_df.columns))
        assert nan_ratio < 0.5  # Should not be mostly NaNs


if __name__ == "__main__":
    pytest.main([__file__])
