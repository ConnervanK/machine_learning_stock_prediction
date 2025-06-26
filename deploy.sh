#!/bin/bash

# Deployment Script for ML Stock Prediction Dashboard
echo "🚀 ML Stock Prediction Dashboard Deployment Script"
echo "=================================================="

# Function to deploy to different platforms
deploy_to_platform() {
    case $1 in
        "streamlit")
            echo "📊 Deploying to Streamlit Community Cloud..."
            echo "1. Make sure your code is pushed to GitHub"
            echo "2. Go to https://share.streamlit.io/"
            echo "3. Connect your GitHub repository"
            echo "4. Set main file: streamlit_dashboard.py"
            echo "5. Click Deploy!"
            ;;
        "docker")
            echo "🐳 Building Docker container..."
            docker build -t ml-stock-dashboard .
            echo "✅ Docker image built successfully!"
            echo "To run locally: docker run -p 8501:8501 ml-stock-dashboard"
            ;;
        "heroku")
            echo "🌐 Deploying to Heroku..."
            if ! command -v heroku &> /dev/null; then
                echo "❌ Heroku CLI not installed. Install from: https://devcenter.heroku.com/articles/heroku-cli"
                exit 1
            fi
            
            # Create Procfile for Heroku
            echo "web: streamlit run streamlit_dashboard.py --server.port \$PORT --server.address 0.0.0.0" > Procfile
            
            heroku create ml-stock-dashboard-$(date +%s)
            git add .
            git commit -m "Deploy to Heroku"
            git push heroku main
            heroku open
            ;;
        "railway")
            echo "🚂 Deploying to Railway..."
            echo "1. Go to https://railway.app/"
            echo "2. Connect your GitHub repository"
            echo "3. Railway will auto-detect and deploy your Streamlit app"
            ;;
        *)
            echo "❌ Unknown platform: $1"
            echo "Available platforms: streamlit, docker, heroku, railway"
            exit 1
            ;;
    esac
}

# Check if platform argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./deploy.sh <platform>"
    echo ""
    echo "Available platforms:"
    echo "  streamlit  - Deploy to Streamlit Community Cloud (FREE)"
    echo "  docker     - Build Docker container"
    echo "  heroku     - Deploy to Heroku (FREE tier available)"
    echo "  railway    - Deploy to Railway (FREE tier available)"
    echo ""
    echo "Example: ./deploy.sh streamlit"
    exit 1
fi

# Deploy to specified platform
deploy_to_platform $1
