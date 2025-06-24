import finnhub
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from newspaper import Article

import os
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# --- Setup Finnhub API ---
finnhub_client = finnhub.Client(api_key='d162ne9r01qhsocmbhs0d162ne9r01qhsocmbhsg')  # Replace with your actual API key

# --- Load FinBERT model and tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# Check for CUDA availability and set the device accordingly
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)  # Move the model to the GPU

label_map = {0: "negative", 1: "neutral", 2: "positive"}

def classify_finbert(text):
    """
    Classify the sentiment of a given text using FinBERT.
    Args:
        text (str): The text to classify.
    Returns:
        tuple: A tuple containing the predicted sentiment label and the probabilities for each class.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)  # Move input tensor to GPU
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    return label_map[predicted_class], probs.squeeze().tolist()

def get_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error fetching article from {url}: {e}")
        return None


import os
import pandas as pd
from datetime import datetime, timedelta

def analyze_article(article_meta):
    url = article_meta.get('url')
    headline = article_meta.get('headline', 'No headline')
    timestamp = article_meta.get('datetime')

    try:
        date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d') if timestamp and timestamp > 0 else "Invalid"
    except Exception as e:
        date_str = "Invalid"

    text_used = ""
    used_full_article = False

    if url:
        full_text = get_article_text(url)
        if full_text and len(full_text) > 20:
            text_used = full_text
            used_full_article = True
        else:
            text_used = headline
    else:
        text_used = headline

    label, probs = classify_finbert(text_used)

    return {
        "date": date_str,
        "negative": round(probs[0], 4),
        "neutral": round(probs[1], 4),
        "positive": round(probs[2], 4),
        "used_full_article": used_full_article,
        "headline": headline
    }


def analyze_and_save_sentiment(symbol, from_date, to_date, output_folder, output_filename, chunk_size_days=5, max_workers=10):
    """
    Fetch and analyze sentiment for a stock symbol over multiple intervals of custom days.
    Saves the combined results to a single Excel file.

    Args:
        symbol (str): Stock symbol.
        from_date (str): Start date in 'YYYY-MM-DD'.
        to_date (str): End date in 'YYYY-MM-DD'.
        output_folder (str): Folder to save results.
        output_filename (str): Output Excel filename (should end with .xlsx).
        chunk_size_days (int): Number of days per interval (default: 5).
        max_workers (int): Number of threads for parallel processing (default: 10).
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    start_date = datetime.strptime(from_date, "%Y-%m-%d")
    end_date = datetime.strptime(to_date, "%Y-%m-%d")
    delta = timedelta(days=chunk_size_days)

    all_rows = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while start_date <= end_date:
            chunk_start = start_date
            chunk_end = min(start_date + delta - timedelta(days=1), end_date)

            print(f"Fetching news for {symbol} from {chunk_start.date()} to {chunk_end.date()}...\n")
            news_data = finnhub_client.company_news(symbol, _from=chunk_start.strftime('%Y-%m-%d'), to=chunk_end.strftime('%Y-%m-%d'))

            # Parallelize sentiment analysis for each article
            results = list(executor.map(analyze_article, news_data))
            all_rows.extend(results)

            start_date += delta  # Move to next chunk

    if all_rows:
        df = pd.DataFrame(all_rows)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Ensure 'date' is datetime
        df = df.sort_values(by="date")  # Sort by date
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')  # Back to string format if needed

        # Filter out again to ensure date range is correct
        df = df[(df['date'] >= from_date) & (df['date'] <= to_date)]
        
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"Saved all results to {output_path}")
    else:
        print("No news articles found in the given date range. No file was saved.")



# analyze_and_save_sentiment(
#     symbol="^GSPC",
#     from_date="2024-01-19",
#     to_date="2025-06-04",
#     output_folder="/home/marcohuy/ETH/AI_assisted/stocks_ml/data",
#     output_filename="SP500_sentiment_190124_040625_gpu_parallel.csv",
#     chunk_size_days=3,  # Custom interval
#     max_workers=10
# )

# analyze_and_save_sentiment(
#     symbol="AAPL",
#     from_date="2024-01-19",
#     to_date="2025-06-04",
#     output_folder="/home/marcohuy/ETH/AI_assisted/stocks_ml/data",
#     output_filename="apple_sentiment_190124_040625_gpu_parallel.csv",
#     chunk_size_days=3,  # Custom interval
#     max_workers=10
# )

# analyze_and_save_sentiment(
#     symbol="MSFT",
#     from_date="2024-01-19",
#     to_date="2025-06-04",
#     output_folder="/home/marcohuy/ETH/AI_assisted/stocks_ml/data",
#     output_filename="microsoft_sentiment_190124_040625_gpu_parallel.csv",
#     chunk_size_days=3,  # Custom interval
#     max_workers=10
# )

# analyze_and_save_sentiment(
#     symbol="TSLA",
#     from_date="2024-01-19",
#     to_date="2025-06-04",
#     output_folder="/home/marcohuy/ETH/AI_assisted/stocks_ml/data",
#     output_filename="tesla_sentiment_190124_040625_gpu_parallel.csv",
#     chunk_size_days=3,  # Custom interval
#     max_workers=10
# )


def create_csv_with_sentiment_month(input_csv, output_csv, filter=True):
    """
    Read an input CSV file with stock data, analyze sentiment for each month, and save results to a new CSV.
    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file.
    """

    df = pd.read_csv(input_csv)
    df['date'] = pd.to_datetime(df['date'])

    # if filter is True, filter out rows where df['used_full_article'] is False
    if filter:
        df = df[df['used_full_article'] == True]
        print(f"Filtered out rows where 'used_full_article' is False.")

    # Remove rows with NaN in sentiment columns to avoid empty means
    df = df.dropna(subset=['negative', 'neutral', 'positive'])

    # Group data by month (use 'MS' for month start to avoid FutureWarning)
    grouped = df.groupby(pd.Grouper(key='date', freq='MS'))

    sentiment_results = []

    for month, group_data in grouped:
        if group_data.empty:
            sentiment_results.append({
                "date": month.strftime('%Y-%m-%d'),
                "negative": "",
                "neutral": "",
                "positive": ""
            })
        else:
            sentiment_results.append({
                "date": month.strftime('%Y-%m-%d'),
                "negative": round(group_data['negative'].mean(), 4),
                "neutral": round(group_data['neutral'].mean(), 4),
                "positive": round(group_data['positive'].mean(), 4)
            })

    # Save results to CSV
    sentiment_df = pd.DataFrame(sentiment_results)
    sentiment_df.to_csv(output_csv, index=False)
    print(f"Sentiment analysis results saved to {output_csv}")


# create_csv_with_sentiment_month(input_csv="/home/marcohuy/ETH/AI_assisted/stocks_ml/data/SP500_sentiment_190124_040625_gpu_parallel.csv",
#                                 output_csv="/home/marcohuy/ETH/AI_assisted/stocks_ml/data/SP500_sentiment_190124_040625_gpu_parallel_month.csv",
#                                 filter=True)

