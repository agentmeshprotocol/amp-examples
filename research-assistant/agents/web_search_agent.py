"""
Web Search Agent for Research Assistant Network

Handles web search, content retrieval, and source validation using multiple search providers.
Integrates with CrewAI for coordinated research workflows.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse
import hashlib

# CrewAI imports
from crewai import Agent, Task
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import GoogleSerperAPIWrapper

# Web scraping and content extraction
import requests
from bs4 import BeautifulSoup
import newspaper
from newspaper import Article

# AMP imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))
from amp_client import AMPClient
from amp_types import Capability, CapabilityConstraints, TransportType
from amp_utils import AMPBuilder


@dataclass
class SearchResult:
    """Represents a search result with metadata."""
    url: str
    title: str
    snippet: str
    content: Optional[str] = None
    source: str = ""
    relevance_score: float = 0.0
    credibility_score: float = 0.0
    publish_date: Optional[datetime] = None
    word_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchQuery:
    """Represents a search query with parameters."""
    query: str
    max_results: int = 10
    time_range: Optional[str] = None  # "1d", "1w", "1m", "1y"
    sources: List[str] = field(default_factory=list)
    language: str = "en"
    region: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)


class WebSearchAgent:
    """Web search agent that discovers and retrieves relevant information."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.WebSearchAgent")
        
        # Initialize search tools
        self.search_tools = self._initialize_search_tools()
        
        # Content extraction
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Research Assistant Bot 1.0 (Educational Use)'
        })
        
        # Caching for efficiency
        self.search_cache: Dict[str, List[SearchResult]] = {}
        self.content_cache: Dict[str, str] = {}
        
        # Quality thresholds
        self.min_word_count = self.config.get("min_word_count", 100)
        self.max_age_days = self.config.get("max_age_days", 365)
        self.credibility_threshold = self.config.get("credibility_threshold", 0.6)
        
        # AMP client
        self.amp_client: Optional[AMPClient] = None
        
        # CrewAI agent
        self.crew_agent = self._create_crew_agent()
    
    def _initialize_search_tools(self) -> Dict[str, Any]:
        """Initialize search tools and APIs."""
        tools = {}
        
        # DuckDuckGo (free, no API key required)
        try:
            tools["duckduckgo"] = DuckDuckGoSearchRun()
            self.logger.info("Initialized DuckDuckGo search")
        except Exception as e:
            self.logger.warning(f"Failed to initialize DuckDuckGo: {e}")
        
        # Google Serper (requires API key)
        serper_key = self.config.get("serper_api_key")
        if serper_key:
            try:
                tools["serper"] = GoogleSerperAPIWrapper(serper_api_key=serper_key)
                self.logger.info("Initialized Google Serper search")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Serper: {e}")
        
        return tools
    
    def _create_crew_agent(self) -> Agent:
        """Create the CrewAI agent for this search agent."""
        return Agent(
            role="Web Search Specialist",
            goal="Find the most relevant, credible, and up-to-date information on any research topic",
            backstory="""You are an expert web researcher with years of experience in 
            information discovery and source validation. You know how to find reliable 
            sources, assess credibility, and extract relevant information efficiently.""",
            tools=[],  # We'll add custom tools
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.1)
        )
    
    async def search_web(self, query: SearchQuery) -> List[SearchResult]:
        """Perform web search across multiple sources."""
        # Check cache first
        cache_key = self._get_cache_key(query)
        if cache_key in self.search_cache:
            self.logger.info(f"Returning cached results for: {query.query}")
            return self.search_cache[cache_key]
        
        all_results = []
        
        # Search with each available tool
        for source_name, tool in self.search_tools.items():
            if query.sources and source_name not in query.sources:
                continue
            
            try:
                results = await self._search_with_tool(tool, source_name, query)
                all_results.extend(results)
                self.logger.info(f"Found {len(results)} results from {source_name}")
            except Exception as e:
                self.logger.error(f"Search failed for {source_name}: {e}")
        
        # Remove duplicates and rank results
        unique_results = self._deduplicate_results(all_results)
        ranked_results = await self._rank_results(unique_results, query)
        
        # Limit results
        final_results = ranked_results[:query.max_results]
        
        # Cache results
        self.search_cache[cache_key] = final_results
        
        self.logger.info(f"Returning {len(final_results)} ranked results")
        return final_results
    
    async def _search_with_tool(self, tool: Any, source: str, query: SearchQuery) -> List[SearchResult]:
        """Search using a specific tool."""
        search_query = self._build_search_query(query)
        
        try:
            if hasattr(tool, 'run'):
                # For tools with run method (like DuckDuckGo)
                raw_results = tool.run(search_query)
                return self._parse_search_results(raw_results, source)
            else:
                # For other API wrappers
                raw_results = tool.search(search_query)
                return self._parse_api_results(raw_results, source)
                
        except Exception as e:
            self.logger.error(f"Search error with {source}: {e}")
            return []
    
    def _build_search_query(self, query: SearchQuery) -> str:
        """Build optimized search query string."""
        search_terms = [query.query]
        
        # Add time constraints
        if query.time_range:
            if query.time_range == "1d":
                search_terms.append("after:1day")
            elif query.time_range == "1w":
                search_terms.append("after:1week")
            elif query.time_range == "1m":
                search_terms.append("after:1month")
            elif query.time_range == "1y":
                search_terms.append("after:1year")
        
        # Add language filter
        if query.language and query.language != "en":
            search_terms.append(f"lang:{query.language}")
        
        return " ".join(search_terms)
    
    def _parse_search_results(self, raw_results: str, source: str) -> List[SearchResult]:
        """Parse search results from string format."""
        results = []
        
        # This is a simplified parser - in production, you'd want more robust parsing
        if not raw_results:
            return results
        
        # Split by common delimiters and extract URLs/titles
        lines = raw_results.split('\\n')
        for line in lines:
            if 'http' in line:
                try:
                    # Extract URL and title (simplified)
                    parts = line.split(' - ')
                    if len(parts) >= 2:
                        url = parts[0].strip()
                        title = parts[1].strip()
                        snippet = parts[2] if len(parts) > 2 else ""
                        
                        result = SearchResult(
                            url=url,
                            title=title,
                            snippet=snippet,
                            source=source,
                            relevance_score=0.5  # Will be calculated later
                        )
                        results.append(result)
                except Exception as e:
                    self.logger.debug(f"Failed to parse line: {line}, error: {e}")
        
        return results
    
    def _parse_api_results(self, raw_results: List[Dict], source: str) -> List[SearchResult]:
        """Parse search results from API response."""
        results = []
        
        for item in raw_results:
            try:
                result = SearchResult(
                    url=item.get("link", ""),
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    source=source,
                    relevance_score=0.5,
                    publish_date=self._parse_date(item.get("date"))
                )
                results.append(result)
            except Exception as e:
                self.logger.debug(f"Failed to parse API result: {item}, error: {e}")
        
        return results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate search results."""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url_normalized = self._normalize_url(result.url)
            if url_normalized not in seen_urls:
                seen_urls.add(url_normalized)
                unique_results.append(result)
        
        return unique_results
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        try:
            parsed = urlparse(url)
            # Remove common tracking parameters
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            return normalized.lower()
        except Exception:
            return url.lower()
    
    async def _rank_results(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Rank search results by relevance and credibility."""
        for result in results:
            # Calculate relevance score
            relevance = self._calculate_relevance(result, query)
            
            # Calculate credibility score
            credibility = self._calculate_credibility(result)
            
            # Combined score (weighted)
            result.relevance_score = relevance
            result.credibility_score = credibility
            
            # Overall score for ranking
            overall_score = (relevance * 0.7) + (credibility * 0.3)
            result.metadata["overall_score"] = overall_score
        
        # Sort by overall score
        return sorted(results, key=lambda r: r.metadata.get("overall_score", 0), reverse=True)
    
    def _calculate_relevance(self, result: SearchResult, query: SearchQuery) -> float:
        """Calculate relevance score for a search result."""
        score = 0.0
        query_terms = query.query.lower().split()
        
        # Title relevance (40% weight)
        title_matches = sum(1 for term in query_terms if term in result.title.lower())
        title_score = (title_matches / len(query_terms)) * 0.4
        
        # Snippet relevance (30% weight)
        snippet_matches = sum(1 for term in query_terms if term in result.snippet.lower())
        snippet_score = (snippet_matches / len(query_terms)) * 0.3
        
        # URL relevance (10% weight)
        url_matches = sum(1 for term in query_terms if term in result.url.lower())
        url_score = (url_matches / len(query_terms)) * 0.1
        
        # Recency bonus (20% weight)
        recency_score = self._calculate_recency_score(result) * 0.2
        
        return min(1.0, title_score + snippet_score + url_score + recency_score)
    
    def _calculate_credibility(self, result: SearchResult) -> float:
        """Calculate credibility score for a source."""
        score = 0.5  # Base score
        
        try:
            domain = urlparse(result.url).netloc.lower()
            
            # High credibility domains
            high_credibility = [
                'wikipedia.org', 'arxiv.org', 'pubmed.ncbi.nlm.nih.gov',
                'nature.com', 'science.org', 'ieee.org', 'acm.org',
                'gov', 'edu', 'bbc.com', 'reuters.com', 'ap.org'
            ]
            
            # Medium credibility domains
            medium_credibility = [
                'nytimes.com', 'washingtonpost.com', 'economist.com',
                'scientificamerican.com', 'nationalgeographic.com'
            ]
            
            # Low credibility indicators
            low_credibility = ['blogspot.', 'wordpress.', 'medium.com']
            
            if any(domain.endswith(hc) for hc in high_credibility):
                score = 0.9
            elif any(hc in domain for hc in high_credibility):
                score = 0.8
            elif any(domain.endswith(mc) for mc in medium_credibility):
                score = 0.7
            elif any(lc in domain for lc in low_credibility):
                score = 0.3
            
            # HTTPS bonus
            if result.url.startswith('https'):
                score += 0.05
            
        except Exception as e:
            self.logger.debug(f"Error calculating credibility for {result.url}: {e}")
        
        return min(1.0, score)
    
    def _calculate_recency_score(self, result: SearchResult) -> float:
        """Calculate recency score (newer is better)."""
        if not result.publish_date:
            return 0.5  # Unknown date gets neutral score
        
        try:
            days_old = (datetime.now() - result.publish_date).days
            
            if days_old <= 1:
                return 1.0
            elif days_old <= 7:
                return 0.9
            elif days_old <= 30:
                return 0.8
            elif days_old <= 90:
                return 0.6
            elif days_old <= 365:
                return 0.4
            else:
                return 0.2
                
        except Exception:
            return 0.5
    
    async def extract_content(self, url: str) -> Optional[str]:
        """Extract full content from a URL."""
        # Check cache first
        if url in self.content_cache:
            return self.content_cache[url]
        
        try:
            # Try with newspaper3k first (better for news articles)
            article = Article(url)
            article.download()
            article.parse()
            
            if article.text and len(article.text) > self.min_word_count:
                content = article.text
            else:
                # Fallback to BeautifulSoup
                content = await self._extract_with_bs4(url)
            
            # Cache the content
            if content:
                self.content_cache[url] = content
            
            return content
            
        except Exception as e:
            self.logger.error(f"Failed to extract content from {url}: {e}")
            return None
    
    async def _extract_with_bs4(self, url: str) -> Optional[str]:
        """Extract content using BeautifulSoup as fallback."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try to find main content
            content_selectors = [
                'article', '[role="main"]', '.content', '.post-content',
                '.entry-content', '.article-body', 'main', '.main'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = "\\n".join(elem.get_text(strip=True) for elem in elements)
                    break
            
            # If no specific content area found, get all text
            if not content:
                content = soup.get_text()
            
            # Clean up the content
            lines = content.split('\\n')
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            content = '\\n'.join(cleaned_lines)
            
            return content if len(content) > self.min_word_count else None
            
        except Exception as e:
            self.logger.error(f"BeautifulSoup extraction failed for {url}: {e}")
            return None
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None
        
        # Common date formats
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%d/%m/%Y",
            "%m/%d/%Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _get_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for search query."""
        key_data = f"{query.query}_{query.max_results}_{query.time_range}_{sorted(query.sources)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    # AMP capability handlers
    async def handle_web_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle web search capability requests."""
        query = SearchQuery(
            query=parameters.get("query", ""),
            max_results=parameters.get("max_results", 10),
            time_range=parameters.get("time_range"),
            sources=parameters.get("sources", []),
            language=parameters.get("language", "en")
        )
        
        results = await self.search_web(query)
        
        return {
            "results": [
                {
                    "url": r.url,
                    "title": r.title,
                    "snippet": r.snippet,
                    "source": r.source,
                    "relevance_score": r.relevance_score,
                    "credibility_score": r.credibility_score,
                    "publish_date": r.publish_date.isoformat() if r.publish_date else None
                }
                for r in results
            ],
            "total_results": len(results),
            "query": query.query
        }
    
    async def handle_content_extraction(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle content extraction capability requests."""
        url = parameters.get("url", "")
        
        if not url:
            return {"error": "URL is required", "content": None}
        
        content = await self.extract_content(url)
        
        return {
            "url": url,
            "content": content,
            "word_count": len(content.split()) if content else 0,
            "success": content is not None
        }
    
    async def start_amp_agent(self, agent_id: str = "web-search-agent",
                             endpoint: str = "http://localhost:8000") -> AMPClient:
        """Start the AMP agent."""
        
        self.amp_client = await (
            AMPBuilder(agent_id, "Web Search Agent")
            .with_framework("crewai")
            .with_transport(TransportType.HTTP, endpoint)
            .add_capability(
                "web-search",
                self.handle_web_search,
                "Search the web for relevant information",
                "search",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 10},
                        "time_range": {"type": "string"},
                        "sources": {"type": "array", "items": {"type": "string"}},
                        "language": {"type": "string", "default": "en"}
                    },
                    "required": ["query"]
                },
                constraints=CapabilityConstraints(response_time_ms=10000)
            )
            .add_capability(
                "content-extraction",
                self.handle_content_extraction,
                "Extract content from web pages",
                "extraction",
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"}
                    },
                    "required": ["url"]
                }
            )
            .build()
        )
        
        return self.amp_client


async def main():
    """Main function for testing the web search agent."""
    logging.basicConfig(level=logging.INFO)
    
    # Create web search agent
    search_agent = WebSearchAgent()
    
    # Start AMP agent
    client = await search_agent.start_amp_agent()
    
    try:
        print("Web Search Agent started. Testing search functionality...")
        
        # Test search
        query = SearchQuery(
            query="artificial intelligence latest developments 2024",
            max_results=5,
            time_range="1m"
        )
        
        results = await search_agent.search_web(query)
        
        print(f"\\nSearch Results for '{query.query}':")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Source: {result.source}")
            print(f"   Relevance: {result.relevance_score:.2f}")
            print(f"   Credibility: {result.credibility_score:.2f}")
            print(f"   Snippet: {result.snippet[:100]}...")
            print()
        
        # Test content extraction
        if results:
            print("Testing content extraction...")
            content = await search_agent.extract_content(results[0].url)
            if content:
                print(f"Extracted {len(content.split())} words from {results[0].url}")
            else:
                print("Failed to extract content")
        
        print("Web Search Agent is running. Press Ctrl+C to stop.")
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())