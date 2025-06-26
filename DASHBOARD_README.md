# 📊 Machine Learning Stock Prediction Dashboard

A comprehensive Streamlit dashboard for analyzing financial markets using machine learning, sentiment analysis, and economic indicators.

## 🚀 Features

### 📈 **Data Overview & Visualization**
- Interactive stock selection (S&P 500, Apple, Microsoft, Tesla, Amazon, NVIDIA)
- Real-time financial data visualization with candlestick charts
- Economic indicators dashboard (GDP, inflation, interest rates, unemployment)
- Key performance metrics and daily changes
- Volume analysis and volatility calculations

### 🔍 **Exploratory Data Analysis**
- Statistical summaries of all variables
- Interactive distribution plots (histograms, box plots)
- Time series analysis and trend identification
- Data quality metrics and completeness reports
- Custom date range filtering

### 📊 **Correlation Analysis**
- Interactive correlation heatmaps
- Cross-correlation analysis between assets
- Lead-lag relationship identification
- Feature importance visualization
- Strongest correlation rankings

### 🤖 **ML Predictions & Forecasting**
- LSTM neural network predictions with uncertainty quantification
- Monte Carlo dropout for prediction confidence intervals
- Configurable model hyperparameters
- Rolling window forecasting
- Real-time model performance metrics
- Backtesting capabilities

### 📰 **Sentiment Analysis**
- News sentiment trends over time
- Sentiment vs stock price correlation analysis
- Sentiment score distributions
- Impact analysis on next-day returns
- Positive/negative news percentage tracking

### ⚙️ **Advanced Model Configuration**
- Interactive hyperparameter tuning
- Feature selection interface
- Model architecture customization
- Training progress visualization
- Model saving and loading capabilities

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Conda or virtualenv (recommended)

### Quick Start

1. **Clone the repository** (if not already done)
   ```bash
   git clone <your-repo-url>
   cd machine_learning_stock_prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_dashboard.txt
   ```

3. **Ensure data is available**
   - Run your data download scripts first
   - Verify the `./data/` folder contains your CSV files

4. **Launch the dashboard**
   ```bash
   # Option 1: Use the launcher script
   ./run_dashboard.sh
   
   # Option 2: Direct streamlit command
   streamlit run streamlit_dashboard.py
   ```

5. **Access the dashboard**
   - Open your browser and go to: `http://localhost:8501`

## 📁 Dashboard Structure

```
streamlit_dashboard.py          # Main dashboard application
requirements_dashboard.txt      # Python dependencies
dashboard_config.ini           # Configuration settings
run_dashboard.sh              # Launch script
├── Tab 1: 📊 Data Overview    # Financial data visualization
├── Tab 2: 🔍 Analysis        # Exploratory data analysis
├── Tab 3: 📈 Correlations    # Correlation analysis
├── Tab 4: 🤖 Predictions     # ML model predictions
├── Tab 5: 📰 Sentiment       # Sentiment analysis
└── Tab 6: ⚙️ Model Config    # Advanced configuration
```

## 🎛️ Dashboard Usage

### **Data Overview Tab**
- Select stocks from the sidebar dropdown
- Choose date ranges for analysis
- View real-time metrics: current price, daily change, volume, volatility
- Analyze price history with interactive candlestick charts
- Monitor economic indicators (GDP, inflation, etc.)

### **Analysis Tab**
- Select datasets for detailed analysis
- View statistical summaries and distributions
- Generate custom distribution plots
- Analyze data quality and completeness

### **Correlations Tab**
- Explore relationships between different variables
- Interactive correlation heatmap
- Identify strongest correlations automatically
- Analyze cross-asset relationships

### **Predictions Tab**
- Configure LSTM model parameters:
  - Sequence length (5-30)
  - Number of epochs (10-100)
  - Hidden dimensions (32-256)
  - Number of layers (1-5)
  - Batch size (16-128)
  - Monte Carlo samples (10-100)
- Generate predictions with uncertainty bands
- View model performance metrics
- Compare actual vs predicted values

### **Sentiment Tab**
- Analyze news sentiment for selected stocks
- View sentiment trends over time
- Examine sentiment score distributions
- Track positive/negative news percentages
- Correlate sentiment with price movements

### **Model Config Tab**
- Advanced hyperparameter tuning
- Feature selection interface
- Model architecture customization
- Save/load model configurations

## 🔧 Configuration

Edit `dashboard_config.ini` to customize:
- Default stocks and date ranges
- Model hyperparameters
- Data paths
- Visualization settings
- Feature selections

## 📊 Key Metrics & Visualizations

### **Financial Metrics**
- Current price and daily changes
- Volume analysis
- Volatility calculations (252-day)
- Price trend analysis

### **Model Performance**
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Prediction confidence intervals
- Backtesting results

### **Sentiment Metrics**
- Average sentiment scores
- Sentiment volatility
- Positive news percentage
- Sentiment-price correlation

## 🚨 Troubleshooting

### **Common Issues**

1. **Data files not found**
   - Ensure you've run the data download scripts
   - Check that `./data/` folder exists and contains CSV files

2. **Module import errors**
   - Verify you're in the correct directory
   - Check that your custom modules are in the `./src/` folder

3. **Performance issues**
   - Reduce the number of epochs for faster predictions
   - Use smaller sequence lengths for quicker processing
   - Clear cache using the "Refresh Data" button

4. **Memory issues**
   - Reduce batch size
   - Use fewer Monte Carlo samples
   - Process smaller date ranges

### **Dependencies**
If you encounter import errors, install missing packages:
```bash
pip install streamlit plotly pandas numpy scikit-learn torch
```

## 🎯 Advanced Features

### **Real-time Updates**
- Auto-refresh capabilities (configurable)
- Live data integration
- Progress bars for long operations

### **Export Capabilities**
- Download predictions as CSV
- Export visualizations as images
- Generate PDF reports (future feature)

### **Customization**
- Modify `dashboard_config.ini` for personalized settings
- Add custom indicators
- Implement additional model architectures

## 🔮 Future Enhancements

- [ ] Real-time data streaming
- [ ] Multiple model comparison
- [ ] Portfolio optimization features
- [ ] Automated trading signals
- [ ] Risk management tools
- [ ] PDF report generation
- [ ] Email alerts and notifications
- [ ] Multi-timeframe analysis
- [ ] Additional sentiment sources
- [ ] Options pricing models

## 📝 Tips for Best Results

1. **Data Quality**: Ensure your data is clean and complete
2. **Model Tuning**: Start with default parameters, then optimize
3. **Feature Selection**: Use domain knowledge to select relevant features
4. **Validation**: Always validate predictions with out-of-sample data
5. **Monitoring**: Regularly monitor model performance and retrain as needed

## 🤝 Contributing

Feel free to contribute improvements:
- Add new visualization features
- Implement additional model architectures
- Enhance the user interface
- Add new data sources
- Improve performance optimizations

## 📄 License

This project is for educational and research purposes. Please ensure compliance with data provider terms of service.

---

**Happy Trading! 📈🚀**
