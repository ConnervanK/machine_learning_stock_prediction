#!/bin/bash

# Setup script for Machine Learning Stock Prediction Dashboard
# This script helps users set up the environment and run the dashboard

echo "🚀 Machine Learning Stock Prediction - Setup Script"
echo "=================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if conda is available
if command_exists conda; then
    echo "✅ Conda found. Setting up conda environment..."
    
    # Ask user if they want to use conda
    read -p "Do you want to use conda environment? (y/n): " use_conda
    
    if [[ $use_conda =~ ^[Yy]$ ]]; then
        echo "📦 Creating conda environment from finance_env_packages.yaml..."
        conda env create -f finance_env_packages.yaml
        
        echo "🔄 Activating environment..."
        conda activate finance_env
        
        echo "✅ Conda environment setup complete!"
    else
        use_conda=false
    fi
else
    echo "⚠️  Conda not found. Using pip installation..."
    use_conda=false
fi

# Pip installation
if [[ $use_conda != true ]]; then
    echo "📦 Installing dependencies with pip..."
    
    # Check if virtual environment should be created
    read -p "Create virtual environment? (recommended) (y/n): " create_venv
    
    if [[ $create_venv =~ ^[Yy]$ ]]; then
        echo "🔧 Creating virtual environment..."
        python -m venv venv
        
        # Activate virtual environment
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
            source venv/Scripts/activate
        else
            source venv/bin/activate
        fi
        
        echo "✅ Virtual environment activated!"
    fi
    
    echo "📥 Installing Python packages..."
    pip install -r requirements.txt
fi

# Download data
echo ""
echo "📊 Setting up data..."
read -p "Download data from ETH Polybox? (y/n): " download_data

if [[ $download_data =~ ^[Yy]$ ]]; then
    echo "⬇️  Downloading data..."
    python src/polybox_download.py
    echo "✅ Data download complete!"
else
    echo "⚠️  Skipping data download. You can download manually later:"
    echo "   python src/polybox_download.py"
    echo "   OR visit: https://polybox.ethz.ch/index.php/s/R99PSQwT9e9CjYy"
fi

# Test installation
echo ""
echo "🧪 Testing installation..."
python minimal_test.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "🚀 To start the dashboard:"
    echo "   streamlit run streamlit_dashboard.py"
    echo ""
    echo "📖 For more information, see:"
    echo "   - README.md for general usage"
    echo "   - DASHBOARD_README.md for detailed dashboard docs"
    echo ""
    echo "🌐 The dashboard will be available at: http://localhost:8501"
else
    echo ""
    echo "❌ Setup encountered issues. Please check the error messages above."
    echo "📞 For help, see the Troubleshooting section in README.md"
fi
