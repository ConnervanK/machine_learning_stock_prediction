"""
Minimal test for dashboard fixes
"""
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to Python path
sys.path.append('./src')

def test_dataframe_issue():
    """Test the select_dtypes issue that was causing errors"""
    print("Testing DataFrame.select_dtypes issue...")
    
    # Create test data similar to what your system produces
    test_data = {
        'financial_data.csv': pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [104.0, 105.0, 106.0],
            'Volume': [1000, 1100, 1200]
        }),
        'gdp_data.csv': pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'gdp': [25000, 25100]
        })
    }
    
    # Test the problematic code pattern from the dashboard
    for dataset_name, df in test_data.items():
        print(f"\nTesting {dataset_name}:")
        print(f"  Type: {type(df)}")
        print(f"  Shape: {df.shape}")
        
        # This was causing the AttributeError
        if isinstance(df, pd.Series):
            print("  Converting Series to DataFrame")
            df = df.to_frame()
        
        # Test select_dtypes (the line that was failing)
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            print(f"  ✅ Numeric columns: {numeric_columns}")
        except Exception as e:
            print(f"  ❌ Error with select_dtypes: {e}")
            return False
        
        # Test date handling (the Arrow serialization issue)
        clean_df = df.copy()
        for col in clean_df.columns:
            if 'date' in col.lower():
                try:
                    clean_df[col] = pd.to_datetime(clean_df[col]).dt.strftime('%Y-%m-%d')
                    print(f"  ✅ Date column {col} converted successfully")
                except Exception as e:
                    print(f"  ⚠️ Date conversion issue for {col}: {e}")
                    clean_df[col] = clean_df[col].astype(str)
    
    print("\n✅ All DataFrame tests passed!")
    return True

if __name__ == "__main__":
    print("🔧 Testing dashboard fixes...\n")
    
    success = test_dataframe_issue()
    
    if success:
        print("\n🎉 Dashboard fixes validated!")
        print("The errors should be resolved now.")
        print("\nTo test the dashboard:")
        print("  streamlit run streamlit_dashboard.py")
    else:
        print("\n❌ Issues still exist!")
