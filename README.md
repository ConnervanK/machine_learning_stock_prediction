# 🚀 Machine Learning Stock Prediction

A comprehensive stock prediction system using machine learning techniques, sentiment analysis, and economic indicators with an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📊 Features

- **LSTM Neural Networks** for stock price prediction with uncertainty quantification
- **Sentiment Analysis** using BERT models on financial news
- **Economic Indicators** integration (GDP, inflation, interest rates, unemployment)
- **Interactive Dashboard** built with Streamlit
- **Correlation Analysis** between different market factors
- **Monte Carlo Dropout** for prediction uncertainty estimation
- **Real-time Data Visualization** with Plotly

## 🎯 Demo

### Dashboard Screenshots
- 📈 Interactive stock price charts with candlestick visualization
- 🔍 Exploratory data analysis with statistical summaries
- 📊 Correlation heatmaps between market indicators
- 🤖 Real-time ML predictions with confidence intervals
- 📰 Sentiment analysis trends over time
- ⚙️ Configurable model hyperparameters

## 🚀 Quick Start

### Option 1: Using Conda (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/machine_learning_stock_prediction.git
   cd machine_learning_stock_prediction
   ```

2. **Create the conda environment**

   ```bash
   conda env create -f finance_env_packages.yaml
   conda activate finance_env
   ```

3. **Download the data**
   ```bash
   python src/polybox_download.py
   # OR manually download from: https://polybox.ethz.ch/index.php/s/R99PSQwT9e9CjYy/download
   ```

4. **Launch the dashboard**
   ```bash
   streamlit run streamlit_dashboard.py
   ```

5. **Open your browser** and go to `http://localhost:8501`

### Option 2: Using pip

1. **Clone and navigate**
   ```bash
   git clone https://github.com/YOUR_USERNAME/machine_learning_stock_prediction.git
   cd machine_learning_stock_prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_dashboard.txt
   ```

3. **Download data and run**
   ```bash
   python src/polybox_download.py
   streamlit run streamlit_dashboard.py
   ```

## 📁 Project Structure

```
machine_learning_stock_prediction/
├── 📊 streamlit_dashboard.py          # Main dashboard application
├── 📋 requirements_dashboard.txt      # Python dependencies
├── 🐳 Dockerfile                     # Container deployment
├── 🔧 finance_env_packages.yaml      # Conda environment
├── 📖 README.md                      # This file
├── 📖 DASHBOARD_README.md            # Detailed dashboard docs
├── 🚀 run_dashboard.sh              # Quick launcher script
├── ⚙️ dashboard_config.ini          # Configuration settings
├── 🧪 test_dashboard.py             # Dashboard validation
├── 📁 src/                          # Source code
│   ├── 🤖 machine_learning_*.py     # ML modules
│   ├── 📰 machine_learning_BERT_articles.py  # Sentiment analysis
│   ├── 📊 machine_learning_plotting.py       # Visualization
│   ├── 🏗️ create_tensor.py          # Data preprocessing
│   └── 📥 polybox_download.py       # Data downloader
├── 📁 data/                         # Data files (auto-generated)
│   ├── 📈 financial_data.csv        # Stock prices
│   ├── 🏛️ economic indicators.csv   # GDP, inflation, etc.
│   └── 📰 sentiment_*.csv           # News sentiment scores
└── 📁 __pycache__/                  # Python cache (auto-generated)
```

## 🎛️ Dashboard Usage

### 📊 **Data Overview Tab**
- Select stocks: S&P 500, Apple, Microsoft, Tesla, Amazon, NVIDIA
- View real-time metrics: price, change, volume, volatility
- Interactive candlestick charts
- Economic indicators visualization

### 🔍 **Analysis Tab**  
- Statistical summaries and data distributions
- Interactive filtering and exploration
- Data quality metrics

### 📈 **Correlations Tab**
- Interactive correlation heatmaps
- Cross-asset relationship analysis
- Feature importance rankings

### 🤖 **Predictions Tab**
- Configure LSTM model parameters
- Generate predictions with uncertainty bands
- Performance metrics (MSE, MAE)
- Real-time model training

### 📰 **Sentiment Tab**
- News sentiment trends over time
- Sentiment vs stock price correlation
- Distribution analysis

### ⚙️ **Model Config Tab**
- Advanced hyperparameter tuning
- Feature selection interface
- Model architecture customization

## 🔧 Configuration

Edit `dashboard_config.ini` to customize:
- Default stocks and date ranges
- Model hyperparameters  
- Data paths and visualization settings
- Feature selections

## 📊 Data Sources

### Financial Data
- Stock prices (OHLC, Volume)
- Economic indicators (GDP, inflation, interest rates, unemployment)

### Sentiment Data  
- Financial news articles processed with BERT
- Sentiment scores correlated with stock movements
- Pre-processed data available via [ETH Polybox](https://polybox.ethz.ch/index.php/s/R99PSQwT9e9CjYy)

## 🧠 Machine Learning Models

### LSTM Neural Networks
- **Architecture**: Multi-layer LSTM with dropout
- **Features**: Multivariate time series (price + economic + sentiment)
- **Uncertainty**: Monte Carlo dropout for confidence intervals
- **Training**: Rolling window with parallel processing

### Sentiment Analysis
- **Model**: BERT-based transformer
- **Input**: Financial news headlines and articles  
- **Output**: Sentiment scores (-1 to +1)
- **Integration**: Combined with price data for predictions

## 🚨 Troubleshooting

### Common Issues

**Data files not found:**
```bash
# Download the data first
python src/polybox_download.py
```

**Module import errors:**
```bash
# Make sure you're in the right directory and environment
conda activate finance_env
# OR
pip install -r requirements_dashboard.txt
```

**Port already in use:**
```bash
# Use a different port
streamlit run streamlit_dashboard.py --server.port 8502
```

**Memory issues:**
```bash
# Reduce model parameters in dashboard
# Or edit dashboard_config.ini
```

## 🎯 Advanced Usage

### Custom Data
1. Add your CSV files to the `data/` folder
2. Update `dashboard_config.ini` 
3. Modify the data loading functions in `src/`

### Model Customization
1. Edit model parameters in the dashboard
2. Or modify `src/machine_learning_training.py`
3. Save/load custom configurations

### Adding New Features
1. Update `src/machine_learning_data.py` for data processing
2. Modify `streamlit_dashboard.py` for visualization
3. Add new correlation analysis in `src/machine_learning_plotting.py`

## 📈 Performance Tips

- **Faster Predictions**: Reduce epochs and sequence length
- **Better Accuracy**: Increase hidden dimensions and layers  
- **Memory Optimization**: Reduce batch size and MC samples
- **Data Processing**: Use the pre-processed Polybox data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/machine_learning_stock_prediction.git

# Install in development mode
conda env create -f finance_env_packages.yaml
conda activate finance_env

# Run tests
python test_dashboard.py
python minimal_test.py
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ETH Zurich** for providing the processed financial data
- **Streamlit** for the amazing dashboard framework
- **Plotly** for interactive visualizations
- **PyTorch** for deep learning capabilities
- **Transformers** for BERT sentiment analysis

## 📞 Support

If you encounter any issues:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the [Dashboard README](DASHBOARD_README.md) for detailed usage
3. Open an issue on GitHub
4. Check that all dependencies are installed correctly

## 🚀 What's Next?

- [ ] Real-time data streaming integration
- [ ] Additional ML models (Random Forest, XGBoost)
- [ ] Portfolio optimization features
- [ ] Mobile-responsive design
- [ ] API endpoints for predictions
- [ ] Automated trading signals
- [ ] Risk management tools

---

**Happy Trading! 📈🚀**

*Star ⭐ this repository if you find it useful!*