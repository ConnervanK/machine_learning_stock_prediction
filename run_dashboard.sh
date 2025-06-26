#!/bin/bash

# Streamlit Dashboard Launcher Script
echo "🚀 Starting Machine Learning Stock Prediction Dashboard..."

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing required packages..."
    pip install -r requirements_dashboard.txt
fi

# Launch the dashboard
echo "📊 Launching dashboard on http://localhost:8501"
streamlit run streamlit_dashboard.py --server.port 8501 --server.address localhost
