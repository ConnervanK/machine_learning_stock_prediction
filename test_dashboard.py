#!/usr/bin/env python3
"""
Test script to validate dashboard fixes
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src directory to Python path
sys.path.append('./src')

def test_data_loading():
    """Test data loading and preprocessing"""
    print("🔧 Testing data loading and preprocessing...")
    
    try:
        import machine_learning_dataloading as mldl
        
        # Test data loading
        data_files = [
            './data/financial_data.csv',
            './data/gdp_data.csv', 
            './data/interest_rate_data.csv',
            './data/inflation_data.csv',
            './data/unemployment_rate_data.csv'
        ]
        
        # Check if files exist
        existing_files = [f for f in data_files if os.path.exists(f)]
        print(f"Found {len(existing_files)} data files: {existing_files}")
        
        if not existing_files:
            print("❌ No data files found!")
            return False
            
        # Test tensor creation
        tensor_data, loaded_data = mldl.create_tensor_from_csvs(existing_files)
        print(f"✅ Tensor created with shape: {tensor_data.shape}")
        
        # Test data cleaning for Streamlit
        print("\n🧹 Testing data cleaning for Streamlit compatibility...")
        cleaned_data = {}
        
        for key, df in loaded_data.items():
            if isinstance(df, pd.DataFrame):
                print(f"Processing {key}: {df.shape}")
                
                # Test select_dtypes (this was causing the error)
                try:
                    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    print(f"  - Numeric columns: {numeric_columns}")
                except Exception as e:
                    print(f"  - ❌ Error with select_dtypes: {e}")
                    return False
                
                # Test date column handling
                clean_df = df.copy()
                for col in clean_df.columns:
                    if 'date' in col.lower():
                        try:
                            clean_df[col] = pd.to_datetime(clean_df[col]).dt.strftime('%Y-%m-%d')
                            print(f"  - ✅ Converted date column: {col}")
                        except Exception as e:
                            print(f"  - ⚠️ Date conversion warning for {col}: {e}")
                            clean_df[col] = clean_df[col].astype(str)
                    elif clean_df[col].dtype == 'object':
                        clean_df[col] = clean_df[col].astype(str)
                
                cleaned_data[key] = clean_df
            else:
                print(f"  - {key} is not a DataFrame: {type(df)}")
                cleaned_data[key] = df
        
        print("✅ Data cleaning completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in data loading test: {e}")
        return False

def test_sentiment_loading():
    """Test sentiment data loading"""
    print("\n📰 Testing sentiment data loading...")
    
    try:
        # Test SP500 sentiment
        sentiment_file = './data/SP500_sentiment_gpu_parallel_filtered.csv'
        if os.path.exists(sentiment_file):
            sentiment_data = pd.read_csv(sentiment_file)
            print(f"✅ Loaded SP500 sentiment data: {sentiment_data.shape}")
            print(f"  - Columns: {list(sentiment_data.columns)}")
            
            # Test date column detection
            date_col = None
            for col in sentiment_data.columns:
                if 'date' in col.lower():
                    date_col = col
                    break
            
            if date_col:
                print(f"  - Found date column: {date_col}")
            else:
                print("  - ⚠️ No date column found")
            
            # Test sentiment score column
            if 'sentiment_score' in sentiment_data.columns:
                avg_sentiment = sentiment_data['sentiment_score'].mean()
                print(f"  - Average sentiment: {avg_sentiment:.3f}")
            else:
                print("  - ⚠️ No sentiment_score column found")
                
            return True
        else:
            print("⚠️ SP500 sentiment file not found")
            return False
            
    except Exception as e:
        print(f"❌ Error in sentiment loading test: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting dashboard validation tests...\n")
    
    # Test 1: Data loading
    test1_passed = test_data_loading()
    
    # Test 2: Sentiment loading
    test2_passed = test_sentiment_loading()
    
    print("\n" + "="*50)
    print("📊 TEST RESULTS:")
    print(f"Data Loading: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Sentiment Loading: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Dashboard should work correctly.")
        print("Run: streamlit run streamlit_dashboard.py")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
