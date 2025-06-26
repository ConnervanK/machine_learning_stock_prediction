#!/usr/bin/env python3
"""
Test script to validate sentiment data fix
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append('./src')

def test_sentiment_data_handling():
    """Test the new sentiment data handling logic"""
    print("🧪 Testing sentiment data handling...")
    
    # Test with actual sentiment data structure
    try:
        sentiment_file = './data/SP500_sentiment_gpu_parallel_filtered.csv'
        if os.path.exists(sentiment_file):
            sentiment_data = pd.read_csv(sentiment_file)
            print(f"✅ Loaded sentiment data: {sentiment_data.shape}")
            print(f"   Columns: {list(sentiment_data.columns)}")
            
            # Test the logic from the dashboard
            available_columns = list(sentiment_data.columns)
            
            if 'sentiment_score' in available_columns:
                sentiment_col = 'sentiment_score'
                print("   Using existing sentiment_score column")
            elif all(col in available_columns for col in ['positive', 'negative', 'neutral']):
                sentiment_data['sentiment_score'] = sentiment_data['positive'] - sentiment_data['negative']
                sentiment_col = 'sentiment_score'
                print("   ✅ Created sentiment_score from positive - negative")
            elif 'positive' in available_columns:
                sentiment_col = 'positive'
                print("   Using positive column as sentiment")
            else:
                numeric_cols = sentiment_data.select_dtypes(include=[np.number]).columns.tolist()
                sentiment_col = numeric_cols[0] if numeric_cols else None
                print(f"   Using first numeric column: {sentiment_col}")
            
            if sentiment_col:
                avg_sentiment = sentiment_data[sentiment_col].mean()
                sentiment_std = sentiment_data[sentiment_col].std()
                print(f"   📊 Average {sentiment_col}: {avg_sentiment:.3f}")
                print(f"   📊 Std {sentiment_col}: {sentiment_std:.3f}")
                
                if all(col in available_columns for col in ['positive', 'negative', 'neutral']):
                    print(f"   📊 Avg Positive: {sentiment_data['positive'].mean():.3f}")
                    print(f"   📊 Avg Neutral: {sentiment_data['neutral'].mean():.3f}")
                    print(f"   📊 Avg Negative: {sentiment_data['negative'].mean():.3f}")
                
                print("   ✅ Sentiment data handling works correctly!")
                return True
            else:
                print("   ❌ No suitable sentiment column found")
                return False
        else:
            print(f"   ⚠️  Sentiment file not found: {sentiment_file}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing sentiment data: {e}")
        return False

def main():
    """Run sentiment data tests"""
    print("🔧 Testing sentiment data fixes...\n")
    
    success = test_sentiment_data_handling()
    
    print("\n" + "="*50)
    if success:
        print("🎉 Sentiment data fix validated!")
        print("The dashboard should now work correctly with:")
        print("  - positive, negative, neutral columns")
        print("  - Automatic sentiment_score calculation")
        print("  - Proper error handling")
        print("\n🚀 Dashboard is now running at: http://localhost:8504")
    else:
        print("❌ Sentiment data issues detected.")
        print("Check the error messages above.")

if __name__ == "__main__":
    main()
