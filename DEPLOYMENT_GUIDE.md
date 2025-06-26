# Streamlit Community Cloud Deployment Guide

## 🚀 Deploy to Streamlit Community Cloud (FREE)

### Prerequisites:
1. GitHub account
2. Your code in a GitHub repository
3. Streamlit Community Cloud account

### Steps:

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - ML Stock Prediction Dashboard"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

2. **Go to Streamlit Community Cloud:**
   - Visit: https://share.streamlit.io/
   - Sign in with your GitHub account

3. **Deploy your app:**
   - Click "New app"
   - Select your repository
   - Set main file path: `streamlit_dashboard.py`
   - Click "Deploy"

4. **Your app will be available at:**
   `https://YOUR_USERNAME-YOUR_REPO_NAME-streamlit-dashboard-xxxxx.streamlit.app`

### Requirements for Streamlit Cloud:
- Make sure your `requirements_dashboard.txt` is in the root directory
- Ensure all file paths are relative (✅ already done in your code)
- Data files should be in the repository or loaded from external sources

---

## 🐳 Alternative: Docker + Cloud Platform

If you need more control, you can containerize your app:

### Dockerfile:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements_dashboard.txt

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_dashboard.py", "--server.address", "0.0.0.0"]
```

### Deploy to:
- **Google Cloud Run**
- **AWS ECS**
- **Azure Container Instances**
- **Heroku**

---

## 🌍 Quick Deploy with ngrok (For Testing)

For immediate testing, you can use ngrok to expose your local server:

```bash
# Install ngrok
# Then run:
ngrok http 8502
```

This gives you a public URL immediately but only works while your computer is on.
