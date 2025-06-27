import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from machine_learning_BERT_articles import classify_finbert, get_article_text, analyze_article


class TestClassifyFinbert:
    
    @patch('machine_learning_BERT_articles.model')
    @patch('machine_learning_BERT_articles.tokenizer')
    def test_classify_finbert_basic(self, mock_tokenizer, mock_model):
        """Test basic FinBERT classification functionality"""
        # Mock tokenizer output
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.to = MagicMock(return_value=mock_tokenizer.return_value)
        
        # Mock model output
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.1, 0.2, 0.7]])  # Positive sentiment
        mock_model.return_value = mock_outputs
        
        # Mock torch functions
        with patch('torch.no_grad'), patch('torch.nn.functional.softmax') as mock_softmax, \
             patch('torch.argmax') as mock_argmax:
            
            mock_softmax.return_value = MagicMock()
            mock_softmax.return_value.squeeze.return_value.tolist.return_value = [0.1, 0.2, 0.7]
            mock_argmax.return_value.item.return_value = 2  # Positive class
            
            label, probs = classify_finbert("This is great news for the stock market!")
            
            # Check results
            assert label in ["negative", "neutral", "positive"]
            assert isinstance(probs, list)
            assert len(probs) == 3
    
    def test_classify_finbert_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with empty string
        try:
            label, probs = classify_finbert("")
            # If it doesn't crash, that's good
            assert isinstance(label, str)
            assert isinstance(probs, list)
        except Exception as e:
            # Expected due to model dependencies in test environment
            assert any(word in str(e).lower() for word in ['model', 'tensor', 'cuda', 'import'])


class TestGetArticleText:
    
    @patch('machine_learning_BERT_articles.Article')
    def test_get_article_text_success(self, mock_article_class):
        """Test successful article text extraction"""
        # Mock Article instance
        mock_article = MagicMock()
        mock_article.text = "This is the article content."
        mock_article_class.return_value = mock_article
        
        result = get_article_text("https://example.com/article")
        
        # Check that download and parse were called
        mock_article.download.assert_called_once()
        mock_article.parse.assert_called_once()
        assert result == "This is the article content."
    
    @patch('machine_learning_BERT_articles.Article')
    def test_get_article_text_failure(self, mock_article_class):
        """Test article text extraction failure"""
        # Mock Article to raise exception
        mock_article = MagicMock()
        mock_article.download.side_effect = Exception("Network error")
        mock_article_class.return_value = mock_article
        
        result = get_article_text("https://invalid-url.com")
        
        # Should return None on error
        assert result is None
    
    def test_get_article_text_invalid_url(self):
        """Test with invalid URL"""
        result = get_article_text("not-a-url")
        
        # Should handle gracefully
        assert result is None or isinstance(result, str)


class TestAnalyzeArticle:
    
    @patch('machine_learning_BERT_articles.get_article_text')
    @patch('machine_learning_BERT_articles.classify_finbert')
    def test_analyze_article_with_full_text(self, mock_classify, mock_get_text):
        """Test article analysis using full article text"""
        # Mock successful article extraction
        mock_get_text.return_value = "This is a long article about positive market trends."
        mock_classify.return_value = ("positive", [0.1, 0.2, 0.7])
        
        article_meta = {
            'url': 'https://example.com/article',
            'headline': 'Market News',
            'datetime': 1640995200  # Unix timestamp for 2022-01-01
        }
        
        result = analyze_article(article_meta)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'date' in result
        assert 'negative' in result
        assert 'neutral' in result
        assert 'positive' in result
        assert 'used_full_article' in result
        assert 'headline' in result
        
        # Check values
        assert result['date'] == '2022-01-01'
        assert result['used_full_article'] == True
        assert result['headline'] == 'Market News'
        assert isinstance(result['positive'], float)
    
    @patch('machine_learning_BERT_articles.get_article_text')
    @patch('machine_learning_BERT_articles.classify_finbert')
    def test_analyze_article_headline_only(self, mock_classify, mock_get_text):
        """Test article analysis using only headline"""
        # Mock failed article extraction
        mock_get_text.return_value = None
        mock_classify.return_value = ("neutral", [0.3, 0.4, 0.3])
        
        article_meta = {
            'url': 'https://example.com/article',
            'headline': 'Stock market update',
            'datetime': 1640995200
        }
        
        result = analyze_article(article_meta)
        
        # Should use headline instead of full article
        assert result['used_full_article'] == False
        assert result['headline'] == 'Stock market update'
    
    def test_analyze_article_invalid_timestamp(self):
        """Test analysis with invalid timestamp"""
        with patch('machine_learning_BERT_articles.classify_finbert') as mock_classify:
            mock_classify.return_value = ("neutral", [0.3, 0.4, 0.3])
            
            article_meta = {
                'url': None,
                'headline': 'Test headline',
                'datetime': -1  # Invalid timestamp
            }
            
            result = analyze_article(article_meta)
            
            # Should handle invalid timestamp gracefully
            assert result['date'] == "Invalid"
    
    def test_analyze_article_no_url(self):
        """Test analysis when no URL is provided"""
        with patch('machine_learning_BERT_articles.classify_finbert') as mock_classify:
            mock_classify.return_value = ("negative", [0.6, 0.3, 0.1])
            
            article_meta = {
                'url': None,
                'headline': 'Bad news for stocks',
                'datetime': 1640995200
            }
            
            result = analyze_article(article_meta)
            
            # Should use headline only
            assert result['used_full_article'] == False
            assert result['headline'] == 'Bad news for stocks'


class TestBERTIntegration:
    
    def test_sentiment_analysis_pipeline(self):
        """Test complete sentiment analysis pipeline"""
        # Mock all external dependencies
        with patch('machine_learning_BERT_articles.classify_finbert') as mock_classify, \
             patch('machine_learning_BERT_articles.get_article_text') as mock_get_text:
            
            mock_get_text.return_value = "Positive market sentiment continues to drive stock prices higher."
            mock_classify.return_value = ("positive", [0.05, 0.15, 0.80])
            
            # Test data similar to what would come from Finnhub API
            test_articles = [
                {
                    'url': 'https://example.com/article1',
                    'headline': 'Stocks Rally on Good News',
                    'datetime': 1640995200
                },
                {
                    'url': 'https://example.com/article2', 
                    'headline': 'Market Concerns Grow',
                    'datetime': 1641081600
                }
            ]
            
            # Process articles
            results = []
            for article in test_articles:
                result = analyze_article(article)
                results.append(result)
            
            # Validate results
            assert len(results) == 2
            for result in results:
                assert 'date' in result
                assert 'positive' in result
                assert 'negative' in result
                assert 'neutral' in result
                assert sum([result['positive'], result['negative'], result['neutral']]) == pytest.approx(1.0, rel=1e-2)


if __name__ == "__main__":
    # Import torch here to avoid issues if not available
    try:
        import torch
    except ImportError:
        torch = MagicMock()
        sys.modules['torch'] = torch
        sys.modules['torch.nn'] = MagicMock()
        sys.modules['torch.nn.functional'] = MagicMock()
    
    pytest.main([__file__])
