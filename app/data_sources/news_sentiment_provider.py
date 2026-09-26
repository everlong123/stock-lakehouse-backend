"""News and Sentiment data provider - scrapes financial news and analyzes sentiment."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from app.core.exceptions import DataSourceError
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class NewsSentimentProvider:
    """Fetch financial news and perform sentiment analysis."""
    
    source_name = "news_sentiment"
    
    # Vietnamese financial news sources
    NEWS_SOURCES = [
        "https://cafef.vn/timeline/{page}.chn",
        "https://s.cafef.vn/TinTuc/{page}.chn",
    ]
    
    # Simple Vietnamese sentiment lexicon (positive/negative words)
    POSITIVE_WORDS = {
        "tăng", "lợi nhuận", "lãi", "tích cực", "khởi sắc", "bứt phá", 
        "tăng trưởng", "lạc quan", "hưởng lợi", "cổ tức", "vượt", "lên",
        "rise", "profit", "growth", "positive", "bullish", "gain", "up"
    }
    
    NEGATIVE_WORDS = {
        "giảm", "thua lỗ", "lỗ", "tiêu cực", "trì trệ", "sụt giảm",
        "rủi ro", "bất lợi", "lao dốc", "xuống", "phá", "giáng",
        "fall", "loss", "decline", "negative", "bearish", "drop", "down"
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
    
    def get_news_for_symbol(self, symbol: str, limit: int = 20) -> pd.DataFrame:
        """Get recent news articles for a symbol."""
        symbol = symbol.upper()
        
        articles = []
        
        # Search CafeF for symbol news
        articles.extend(self._search_cafef_news(symbol, limit))
        
        if not articles:
            logger.warning("No news found for %s", symbol)
            return pd.DataFrame()
        
        # Add sentiment scores
        for article in articles:
            article["sentiment"] = self._analyze_sentiment(article.get("title", "") + " " + article.get("content", ""))
        
        frame = pd.DataFrame(articles)
        frame["source"] = self.source_name
        frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")
        
        return frame
    
    def _search_cafef_news(self, symbol: str, limit: int) -> list[dict]:
        """Search CafeF for news articles."""
        articles = []
        
        try:
            # Try different search patterns
            search_urls = [
                f"https://cafef.vn/tim-kiem.chn?k={symbol}",
                f"https://s.cafef.vn/TinTuc/{symbol}/1",
            ]
            
            for url in search_urls:
                if len(articles) >= limit:
                    break
                    
                response = self.session.get(url, timeout=15)
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Find news items
                news_items = soup.find_all("div", {"class": "news-item"}) or \
                             soup.find_all("li", {"class": "news"}) or \
                             soup.find_all("article")
                
                for item in news_items[:limit]:
                    try:
                        title_elem = item.find("h3") or item.find("a") or item.find("h2")
                        title = title_elem.get_text(strip=True) if title_elem else ""
                        
                        # Skip if doesn't contain symbol
                        if symbol not in title.upper():
                            continue
                        
                        link_elem = item.find("a")
                        link = link_elem.get("href", "") if link_elem else ""
                        if link and not link.startswith("http"):
                            link = f"https://cafef.vn{link}"
                        
                        date_elem = item.find("span", {"class": "date"}) or item.find("time")
                        date_str = date_elem.get_text(strip=True) if date_elem else ""
                        pub_date = self._parse_date(date_str)
                        
                        articles.append({
                            "symbol": symbol,
                            "title": title,
                            "content": "",
                            "url": link,
                            "published_date": pub_date,
                            "source": "cafef",
                        })
                    except Exception:
                        continue
                        
        except Exception as exc:
            logger.warning("CafeF news search failed for %s: %s", symbol, exc)
        
        return articles[:limit]
    
    def get_market_news(self, limit: int = 30) -> pd.DataFrame:
        """Get general market news."""
        articles = []
        
        try:
            url = "https://s.cafef.vn/TinTuc.chn"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                
                news_items = soup.find_all("div", {"class": "item"}) or \
                             soup.find_all("li", {"class": "clearfix"})
                
                for item in news_items[:limit]:
                    try:
                        title_elem = item.find("h3") or item.find("h4") or item.find("a")
                        title = title_elem.get_text(strip=True) if title_elem else ""
                        
                        link_elem = item.find("a")
                        link = link_elem.get("href", "") if link_elem else ""
                        if link and not link.startswith("http"):
                            link = f"https://cafef.vn{link}"
                        
                        articles.append({
                            "symbol": None,
                            "title": title,
                            "content": "",
                            "url": link,
                            "published_date": datetime.now(timezone.utc),
                            "source": "cafef",
                        })
                    except Exception:
                        continue
                        
        except Exception as exc:
            logger.warning("Market news fetch failed: %s", exc)
        
        # Add sentiment
        for article in articles:
            article["sentiment"] = self._analyze_sentiment(article.get("title", ""))
        
        frame = pd.DataFrame(articles)
        if not frame.empty:
            frame["source"] = self.source_name
            frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")
        
        return frame
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple lexicon-based sentiment analysis."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in self.POSITIVE_WORDS)
        negative_count = sum(1 for word in words if word in self.NEGATIVE_WORDS)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        # Return sentiment score from -1 (negative) to 1 (positive)
        return (positive_count - negative_count) / total
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse Vietnamese date string."""
        if not date_str:
            return datetime.now(timezone.utc)
        
        formats = [
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        
        return datetime.now(timezone.utc)
    
    def get_sentiment_summary(self, symbol: str, days: int = 7) -> dict[str, Any]:
        """Get aggregated sentiment summary for a symbol."""
        news = self.get_news_for_symbol(symbol, limit=50)
        
        if news.empty:
            return {
                "symbol": symbol,
                "total_articles": 0,
                "avg_sentiment": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "trend": "neutral",
            }
        
        # Filter to recent days
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        news = news[news["published_date"] >= cutoff]
        
        avg_sentiment = news["sentiment"].mean()
        positive = len(news[news["sentiment"] > 0.1])
        negative = len(news[news["sentiment"] < -0.1])
        neutral = len(news) - positive - negative
        
        if avg_sentiment > 0.2:
            trend = "positive"
        elif avg_sentiment < -0.2:
            trend = "negative"
        else:
            trend = "neutral"
        
        return {
            "symbol": symbol,
            "total_articles": len(news),
            "avg_sentiment": float(avg_sentiment),
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,
            "trend": trend,
            "period_days": days,
        }
