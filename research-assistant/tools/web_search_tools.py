"""
Web Search Tools for Research Assistant Network

Specialized tools for web search, content retrieval, and source validation.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse, quote_plus
import json
import re

# Web scraping
import requests
from bs4 import BeautifulSoup
import newspaper
from newspaper import Article

# Search APIs
from duckduckgo_search import DDGS
try:
    from googleapiclient.discovery import build
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


@dataclass
class SearchResult:
    """Enhanced search result with metadata."""
    url: str
    title: str
    snippet: str
    content: Optional[str] = None
    source_name: str = ""
    domain: str = ""
    relevance_score: float = 0.0
    credibility_score: float = 0.0
    publish_date: Optional[datetime] = None
    word_count: int = 0
    language: str = "en"
    content_type: str = "article"
    metadata: Dict[str, Any] = None


class WebSearchTools:
    """Comprehensive web search and content tools."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.WebSearchTools")
        
        # API keys
        self.google_api_key = self.config.get("google_api_key")
        self.google_cse_id = self.config.get("google_cse_id")
        self.serper_api_key = self.config.get("serper_api_key")
        
        # HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Research Assistant Bot 1.0 (Educational/Research Use)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        # Credible domains for scoring
        self.credible_domains = self._load_credible_domains()
        
        # Content extraction cache
        self.content_cache: Dict[str, str] = {}
    
    def _load_credible_domains(self) -> Dict[str, float]:
        """Load credible domains with credibility scores."""
        return {
            # High credibility news
            "reuters.com": 0.95,
            "ap.org": 0.95,
            "bbc.com": 0.90,
            "npr.org": 0.88,
            "pbs.org": 0.87,
            
            # Academic and research
            "pubmed.ncbi.nlm.nih.gov": 0.98,
            "arxiv.org": 0.92,
            "nature.com": 0.95,
            "science.org": 0.95,
            "ieee.org": 0.93,
            "acm.org": 0.92,
            "springer.com": 0.88,
            "wiley.com": 0.87,
            
            # Government sources
            "cdc.gov": 0.92,
            "who.int": 0.90,
            "fda.gov": 0.90,
            "nih.gov": 0.92,
            "census.gov": 0.95,
            "energy.gov": 0.88,
            "epa.gov": 0.88,
            
            # Reference and encyclopedic
            "wikipedia.org": 0.75,
            "britannica.com": 0.88,
            "merriam-webster.com": 0.85,
            
            # Fact-checking
            "snopes.com": 0.85,
            "factcheck.org": 0.88,
            "politifact.com": 0.82,
            
            # Quality news outlets
            "nytimes.com": 0.82,
            "washingtonpost.com": 0.82,
            "wsj.com": 0.85,
            "economist.com": 0.87,
            "guardian.com": 0.80,
            
            # Tech and science
            "scientificamerican.com": 0.87,
            "nationalgeographic.com": 0.85,
            "smithsonianmag.com": 0.83,
            "newscientist.com": 0.82,
        }
    
    async def search_duckduckgo(self, query: str, max_results: int = 10, 
                               region: str = "us-en", time_range: Optional[str] = None) -> List[SearchResult]:
        """Search using DuckDuckGo (free, no API key required)."""
        try:
            results = []
            
            # Prepare search parameters
            search_params = {
                "keywords": query,
                "region": region,
                "safesearch": "moderate",
                "max_results": max_results
            }
            
            if time_range:
                search_params["timelimit"] = time_range
            
            # Perform search
            with DDGS() as ddgs:
                search_results = list(ddgs.text(**search_params))
            
            # Process results
            for i, result in enumerate(search_results):
                search_result = SearchResult(
                    url=result.get("href", ""),
                    title=result.get("title", ""),
                    snippet=result.get("body", ""),
                    source_name="DuckDuckGo",
                    domain=self._extract_domain(result.get("href", "")),
                    relevance_score=max(0, 1.0 - (i * 0.05)),  # Decreasing relevance
                    metadata={"search_rank": i + 1, "search_engine": "duckduckgo"}
                )
                
                # Calculate credibility score
                search_result.credibility_score = self._calculate_credibility(search_result.domain)
                
                results.append(search_result)
            
            self.logger.info(f"DuckDuckGo search found {len(results)} results for: {query}")
            return results
            
        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    async def search_google(self, query: str, max_results: int = 10, 
                           site_restrict: Optional[str] = None) -> List[SearchResult]:
        """Search using Google Custom Search API (requires API key)."""
        if not self.google_api_key or not self.google_cse_id:
            self.logger.warning("Google API credentials not configured")
            return []
        
        if not GOOGLE_AVAILABLE:
            self.logger.warning("Google API client not available")
            return []
        
        try:
            service = build("customsearch", "v1", developerKey=self.google_api_key)
            
            search_params = {
                "q": query,
                "cx": self.google_cse_id,
                "num": min(max_results, 10)  # Google API limit
            }
            
            if site_restrict:
                search_params["siteSearch"] = site_restrict
            
            result = service.cse().list(**search_params).execute()
            
            results = []
            items = result.get("items", [])
            
            for i, item in enumerate(items):
                search_result = SearchResult(
                    url=item.get("link", ""),
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    source_name="Google",
                    domain=self._extract_domain(item.get("link", "")),
                    relevance_score=max(0, 1.0 - (i * 0.05)),
                    metadata={
                        "search_rank": i + 1,
                        "search_engine": "google",
                        "display_link": item.get("displayLink", "")
                    }
                )
                
                search_result.credibility_score = self._calculate_credibility(search_result.domain)
                results.append(search_result)
            
            self.logger.info(f"Google search found {len(results)} results for: {query}")
            return results
            
        except Exception as e:
            self.logger.error(f"Google search failed: {e}")
            return []
    
    async def search_news_api(self, query: str, max_results: int = 10, 
                             language: str = "en", sort_by: str = "relevancy") -> List[SearchResult]:
        """Search news articles using News API (requires API key)."""
        news_api_key = self.config.get("news_api_key")
        if not news_api_key:
            self.logger.warning("News API key not configured")
            return []
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "language": language,
                "sortBy": sort_by,
                "pageSize": min(max_results, 100),
                "apiKey": news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get("articles", [])
            
            results = []
            for i, article in enumerate(articles):
                search_result = SearchResult(
                    url=article.get("url", ""),
                    title=article.get("title", ""),
                    snippet=article.get("description", ""),
                    source_name=article.get("source", {}).get("name", "News API"),
                    domain=self._extract_domain(article.get("url", "")),
                    relevance_score=max(0, 1.0 - (i * 0.03)),
                    publish_date=self._parse_date(article.get("publishedAt")),
                    content_type="news",
                    metadata={
                        "search_rank": i + 1,
                        "search_engine": "news_api",
                        "author": article.get("author"),
                        "source_name": article.get("source", {}).get("name")
                    }
                )
                
                search_result.credibility_score = self._calculate_credibility(search_result.domain)
                results.append(search_result)
            
            self.logger.info(f"News API search found {len(results)} results for: {query}")
            return results
            
        except Exception as e:
            self.logger.error(f"News API search failed: {e}")
            return []
    
    async def search_academic_sources(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search academic sources including arXiv, PubMed, etc."""
        all_results = []
        
        # Search arXiv
        arxiv_results = await self._search_arxiv(query, max_results // 2)
        all_results.extend(arxiv_results)
        
        # Search PubMed (via Entrez API)
        pubmed_results = await self._search_pubmed(query, max_results // 2)
        all_results.extend(pubmed_results)
        
        # Search Google Scholar (if available)
        scholar_results = await self._search_google_scholar(query, max_results // 3)
        all_results.extend(scholar_results)
        
        return all_results[:max_results]
    
    async def _search_arxiv(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search arXiv preprints."""
        try:
            url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.content)
            
            results = []
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")
            
            for i, entry in enumerate(entries):
                title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                id_elem = entry.find("{http://www.w3.org/2005/Atom}id")
                published_elem = entry.find("{http://www.w3.org/2005/Atom}published")
                
                if title_elem is not None and id_elem is not None:
                    search_result = SearchResult(
                        url=id_elem.text,
                        title=title_elem.text.strip(),
                        snippet=summary_elem.text.strip() if summary_elem is not None else "",
                        source_name="arXiv",
                        domain="arxiv.org",
                        relevance_score=max(0, 1.0 - (i * 0.05)),
                        credibility_score=0.92,
                        publish_date=self._parse_date(published_elem.text if published_elem is not None else None),
                        content_type="academic",
                        metadata={
                            "search_rank": i + 1,
                            "search_engine": "arxiv",
                            "source_type": "preprint"
                        }
                    )
                    results.append(search_result)
            
            self.logger.info(f"arXiv search found {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"arXiv search failed: {e}")
            return []
    
    async def _search_pubmed(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search PubMed for medical/biological research."""
        try:
            # Search PubMed via Entrez API
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance"
            }
            
            search_response = requests.get(search_url, params=search_params, timeout=10)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            ids = search_data.get("esearchresult", {}).get("idlist", [])
            
            if not ids:
                return []
            
            # Get detailed information for found articles
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "json"
            }
            
            fetch_response = requests.get(fetch_url, params=fetch_params, timeout=10)
            fetch_response.raise_for_status()
            fetch_data = fetch_response.json()
            
            results = []
            articles = fetch_data.get("result", {})
            
            for i, pmid in enumerate(ids):
                article = articles.get(pmid, {})
                
                if article:
                    search_result = SearchResult(
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        title=article.get("title", ""),
                        snippet=article.get("abstract", "")[:300] + "...",
                        source_name="PubMed",
                        domain="pubmed.ncbi.nlm.nih.gov",
                        relevance_score=max(0, 1.0 - (i * 0.05)),
                        credibility_score=0.98,
                        publish_date=self._parse_pubmed_date(article.get("pubdate")),
                        content_type="academic",
                        metadata={
                            "search_rank": i + 1,
                            "search_engine": "pubmed",
                            "pmid": pmid,
                            "journal": article.get("source", ""),
                            "authors": article.get("authors", [])
                        }
                    )
                    results.append(search_result)
            
            self.logger.info(f"PubMed search found {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"PubMed search failed: {e}")
            return []
    
    async def _search_google_scholar(self, query: str, max_results: int = 3) -> List[SearchResult]:
        """Search Google Scholar (limited due to rate limiting)."""
        # Note: This is a simplified implementation
        # In production, you'd use a proper API or library like scholarly
        try:
            # Use regular Google search with site restriction
            scholar_query = f"site:scholar.google.com {query}"
            results = await self.search_google(scholar_query, max_results)
            
            # Update metadata to indicate Scholar source
            for result in results:
                result.source_name = "Google Scholar"
                result.content_type = "academic"
                result.credibility_score = min(0.85, result.credibility_score + 0.1)
                result.metadata["source_type"] = "academic_search"
            
            return results
            
        except Exception as e:
            self.logger.error(f"Google Scholar search failed: {e}")
            return []
    
    async def extract_content(self, url: str, method: str = "auto") -> Optional[str]:
        """Extract full text content from a URL."""
        # Check cache first
        if url in self.content_cache:
            return self.content_cache[url]
        
        content = None
        
        if method == "auto":
            # Try newspaper3k first, then fallback to BeautifulSoup
            content = await self._extract_with_newspaper(url)
            if not content or len(content) < 100:
                content = await self._extract_with_beautifulsoup(url)
        elif method == "newspaper":
            content = await self._extract_with_newspaper(url)
        elif method == "beautifulsoup":
            content = await self._extract_with_beautifulsoup(url)
        
        # Cache successful extractions
        if content and len(content) > 100:
            self.content_cache[url] = content
        
        return content
    
    async def _extract_with_newspaper(self, url: str) -> Optional[str]:
        """Extract content using newspaper3k."""
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            if article.text and len(article.text) > 100:
                return article.text
            return None
            
        except Exception as e:
            self.logger.debug(f"Newspaper extraction failed for {url}: {e}")
            return None
    
    async def _extract_with_beautifulsoup(self, url: str) -> Optional[str]:
        """Extract content using BeautifulSoup."""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()
            
            # Try to find main content
            content_selectors = [
                'article', '[role="main"]', '.content', '.post-content',
                '.entry-content', '.article-body', 'main', '.main',
                '.post', '.article', '#content', '#main'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = "\n".join(elem.get_text(strip=True) for elem in elements)
                    break
            
            # If no specific content area found, get all text
            if not content:
                content = soup.get_text()
            
            # Clean up the content
            lines = content.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and len(line) > 20:  # Filter out very short lines
                    cleaned_lines.append(line)
            
            content = '\n'.join(cleaned_lines)
            
            return content if len(content) > 100 else None
            
        except Exception as e:
            self.logger.debug(f"BeautifulSoup extraction failed for {url}: {e}")
            return None
    
    def validate_sources(self, sources: List[SearchResult]) -> List[SearchResult]:
        """Validate and filter sources based on credibility and relevance."""
        validated_sources = []
        
        for source in sources:
            # Skip sources with very low credibility
            if source.credibility_score < 0.3:
                continue
            
            # Skip sources with missing essential information
            if not source.url or not source.title:
                continue
            
            # Check for duplicate URLs
            if any(vs.url == source.url for vs in validated_sources):
                continue
            
            # Validate URL format
            if not self._is_valid_url(source.url):
                continue
            
            validated_sources.append(source)
        
        # Sort by combined credibility and relevance score
        validated_sources.sort(
            key=lambda x: (x.credibility_score * 0.6 + x.relevance_score * 0.4),
            reverse=True
        )
        
        return validated_sources
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception:
            return ""
    
    def _calculate_credibility(self, domain: str) -> float:
        """Calculate credibility score for a domain."""
        if domain in self.credible_domains:
            return self.credible_domains[domain]
        
        # Calculate based on domain characteristics
        score = 0.5  # Base score
        
        if domain.endswith('.gov'):
            score += 0.3
        elif domain.endswith('.edu'):
            score += 0.25
        elif domain.endswith('.org'):
            score += 0.1
        
        # Quality indicators
        quality_indicators = [
            'university', 'institute', 'academy', 'college',
            'journal', 'research', 'science', 'medical'
        ]
        
        for indicator in quality_indicators:
            if indicator in domain:
                score += 0.1
                break
        
        # Low quality indicators
        low_quality = ['blog', 'wordpress', 'blogspot', 'tumblr']
        for indicator in low_quality:
            if indicator in domain:
                score -= 0.2
                break
        
        return max(0.0, min(1.0, score))
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None
        
        # Common date formats
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _parse_pubmed_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse PubMed date format."""
        if not date_str:
            return None
        
        try:
            # PubMed dates are often in format "2024" or "2024 Jan" or "2024 Jan 15"
            parts = date_str.split()
            year = int(parts[0])
            
            if len(parts) == 1:
                return datetime(year, 1, 1)
            elif len(parts) == 2:
                month_map = {
                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                }
                month = month_map.get(parts[1], 1)
                return datetime(year, month, 1)
            elif len(parts) == 3:
                month_map = {
                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                }
                month = month_map.get(parts[1], 1)
                day = int(parts[2])
                return datetime(year, month, day)
        except Exception:
            pass
        
        return None
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and accessible."""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False
    
    async def get_source_metadata(self, url: str) -> Dict[str, Any]:
        """Get comprehensive metadata for a source."""
        try:
            response = self.session.head(url, timeout=10, allow_redirects=True)
            
            metadata = {
                "url": url,
                "final_url": response.url,
                "status_code": response.status_code,
                "domain": self._extract_domain(url),
                "credibility_score": self._calculate_credibility(self._extract_domain(url)),
                "content_type": response.headers.get("content-type", ""),
                "content_length": response.headers.get("content-length"),
                "last_modified": response.headers.get("last-modified"),
                "server": response.headers.get("server"),
                "accessible": response.status_code == 200
            }
            
            return metadata
            
        except Exception as e:
            return {
                "url": url,
                "error": str(e),
                "accessible": False,
                "credibility_score": 0.0
            }


# Utility functions for common search operations
async def multi_source_search(query: str, max_results_per_source: int = 5, 
                            include_academic: bool = True) -> List[SearchResult]:
    """Perform search across multiple sources and combine results."""
    search_tools = WebSearchTools()
    
    all_results = []
    
    # DuckDuckGo search
    ddg_results = await search_tools.search_duckduckgo(query, max_results_per_source)
    all_results.extend(ddg_results)
    
    # Google search (if configured)
    google_results = await search_tools.search_google(query, max_results_per_source)
    all_results.extend(google_results)
    
    # News search
    news_results = await search_tools.search_news_api(query, max_results_per_source)
    all_results.extend(news_results)
    
    # Academic search (if requested)
    if include_academic:
        academic_results = await search_tools.search_academic_sources(query, max_results_per_source)
        all_results.extend(academic_results)
    
    # Validate and deduplicate
    validated_results = search_tools.validate_sources(all_results)
    
    return validated_results


async def extract_content_batch(urls: List[str], max_concurrent: int = 5) -> Dict[str, Optional[str]]:
    """Extract content from multiple URLs concurrently."""
    search_tools = WebSearchTools()
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def extract_single(url: str) -> Tuple[str, Optional[str]]:
        async with semaphore:
            content = await search_tools.extract_content(url)
            return url, content
    
    # Create tasks for all URLs
    tasks = [extract_single(url) for url in urls]
    
    # Execute concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    content_map = {}
    for result in results:
        if isinstance(result, Exception):
            continue
        url, content = result
        content_map[url] = content
    
    return content_map


if __name__ == "__main__":
    async def test_search_tools():
        """Test the search tools."""
        logging.basicConfig(level=logging.INFO)
        
        search_tools = WebSearchTools()
        
        # Test DuckDuckGo search
        print("Testing DuckDuckGo search...")
        results = await search_tools.search_duckduckgo("artificial intelligence research 2024", 5)
        
        for i, result in enumerate(results):
            print(f"{i+1}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Domain: {result.domain}")
            print(f"   Credibility: {result.credibility_score:.2f}")
            print(f"   Snippet: {result.snippet[:100]}...")
            print()
        
        # Test content extraction
        if results:
            print("Testing content extraction...")
            content = await search_tools.extract_content(results[0].url)
            if content:
                print(f"Extracted {len(content)} characters from {results[0].url}")
                print(f"First 200 chars: {content[:200]}...")
            else:
                print("Failed to extract content")
    
    asyncio.run(test_search_tools())