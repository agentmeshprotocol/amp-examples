"""
Fact Checker Agent for Research Assistant Network

Handles fact verification, claim validation, and source credibility assessment.
Integrates with CrewAI for coordinated research workflows.
"""

import asyncio
import logging
import re
import aiohttp
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
import hashlib

# NLP and text processing
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# CrewAI imports
from crewai import Agent, Task
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchRun

# Web scraping
import requests
from bs4 import BeautifulSoup

# AMP imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))
from amp_client import AMPClient
from amp_types import Capability, CapabilityConstraints, TransportType
from amp_utils import AMPBuilder


@dataclass
class Claim:
    """Represents a factual claim to be verified."""
    text: str
    source: Optional[str] = None
    claim_type: str = "general"  # general, statistical, temporal, causal
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Represents the result of fact verification."""
    claim: Claim
    verdict: str  # "supported", "contradicted", "unverified", "insufficient_evidence"
    confidence_score: float  # 0.0 to 1.0
    supporting_sources: List[Dict[str, Any]] = field(default_factory=list)
    contradicting_sources: List[Dict[str, Any]] = field(default_factory=list)
    verification_details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SourceCredibility:
    """Represents source credibility assessment."""
    url: str
    domain: str
    credibility_score: float  # 0.0 to 1.0
    credibility_factors: Dict[str, Any]
    last_updated: datetime = field(default_factory=datetime.now)


class FactChecker:
    """Fact checker agent that verifies claims and validates information."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.FactChecker")
        
        # Initialize NLP models
        self._initialize_nlp_models()
        
        # Known credible sources
        self.credible_sources = self._load_credible_sources()
        
        # Verification cache
        self.verification_cache: Dict[str, VerificationResult] = {}
        self.source_credibility_cache: Dict[str, SourceCredibility] = {}
        
        # HTTP session for web requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Research Assistant Fact Checker 1.0 (Educational Use)'
        })
        
        # Configuration
        self.min_sources_for_verification = self.config.get("min_sources_for_verification", 2)
        self.max_search_results = self.config.get("max_search_results", 10)
        self.verification_timeout = self.config.get("verification_timeout", 30)
        
        # Search tools
        self.search_tool = DuckDuckGoSearchRun()
        
        # AMP client
        self.amp_client: Optional[AMPClient] = None
        
        # CrewAI agent
        self.crew_agent = self._create_crew_agent()
    
    def _initialize_nlp_models(self):
        """Initialize NLP models for semantic analysis."""
        try:
            # Load spaCy model for NER and text processing
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("Loaded spaCy English model")
        except OSError:
            self.logger.warning("spaCy English model not found")
            self.nlp = None
        
        try:
            # Initialize sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Loaded SentenceTransformer model")
        except Exception as e:
            self.logger.warning(f"Failed to load SentenceTransformer: {e}")
            self.sentence_model = None
    
    def _load_credible_sources(self) -> Dict[str, Dict[str, Any]]:
        """Load known credible sources and their characteristics."""
        return {
            # News organizations
            "reuters.com": {"credibility": 0.95, "bias": "center", "type": "news"},
            "ap.org": {"credibility": 0.95, "bias": "center", "type": "news"},
            "bbc.com": {"credibility": 0.90, "bias": "center-left", "type": "news"},
            "npr.org": {"credibility": 0.88, "bias": "center-left", "type": "news"},
            
            # Academic and research
            "pubmed.ncbi.nlm.nih.gov": {"credibility": 0.98, "bias": "center", "type": "academic"},
            "arxiv.org": {"credibility": 0.92, "bias": "center", "type": "academic"},
            "nature.com": {"credibility": 0.95, "bias": "center", "type": "academic"},
            "science.org": {"credibility": 0.95, "bias": "center", "type": "academic"},
            "ieee.org": {"credibility": 0.93, "bias": "center", "type": "academic"},
            
            # Government sources
            "cdc.gov": {"credibility": 0.92, "bias": "center", "type": "government"},
            "who.int": {"credibility": 0.90, "bias": "center", "type": "government"},
            "fda.gov": {"credibility": 0.90, "bias": "center", "type": "government"},
            "census.gov": {"credibility": 0.95, "bias": "center", "type": "government"},
            
            # Reference sources
            "wikipedia.org": {"credibility": 0.75, "bias": "center", "type": "reference"},
            "britannica.com": {"credibility": 0.88, "bias": "center", "type": "reference"},
            
            # Fact-checking organizations
            "snopes.com": {"credibility": 0.85, "bias": "center", "type": "fact-check"},
            "factcheck.org": {"credibility": 0.88, "bias": "center", "type": "fact-check"},
            "politifact.com": {"credibility": 0.82, "bias": "center-left", "type": "fact-check"}
        }
    
    def _create_crew_agent(self) -> Agent:
        """Create the CrewAI agent for fact checking."""
        return Agent(
            role="Fact Verification Specialist",
            goal="Verify factual claims with high accuracy using multiple credible sources",
            backstory="""You are an expert fact-checker with extensive experience in 
            information verification, source evaluation, and cross-referencing. You excel 
            at identifying reliable sources, detecting misinformation, and providing 
            evidence-based assessments of factual claims.""",
            tools=[],  # We'll add custom tools
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.1)
        )
    
    async def verify_claim(self, claim: Claim) -> VerificationResult:
        """Verify a factual claim using multiple sources."""
        # Check cache first
        cache_key = self._get_cache_key(claim)
        if cache_key in self.verification_cache:
            self.logger.info(f"Returning cached verification for: {claim.text[:50]}...")
            return self.verification_cache[cache_key]
        
        self.logger.info(f"Starting verification of claim: {claim.text[:100]}...")
        
        # Initialize result
        result = VerificationResult(
            claim=claim,
            verdict="unverified",
            confidence_score=0.0,
            verification_details={
                "search_queries": [],
                "sources_checked": 0,
                "verification_method": "multi_source_comparison"
            }
        )
        
        try:
            # Extract key information from claim
            key_info = self._extract_claim_information(claim)
            
            # Generate search queries
            search_queries = self._generate_search_queries(claim, key_info)
            result.verification_details["search_queries"] = search_queries
            
            # Search for supporting/contradicting evidence
            all_sources = []
            for query in search_queries:
                sources = await self._search_for_evidence(query)
                all_sources.extend(sources)
            
            # Remove duplicates and assess credibility
            unique_sources = self._deduplicate_sources(all_sources)
            credible_sources = await self._assess_source_credibility(unique_sources)
            
            result.verification_details["sources_checked"] = len(credible_sources)
            
            # Extract relevant content from sources
            source_contents = await self._extract_source_contents(credible_sources)
            
            # Compare claim against source contents
            verification_analysis = await self._analyze_claim_against_sources(
                claim, source_contents
            )
            
            # Determine verdict and confidence
            result.verdict = verification_analysis["verdict"]
            result.confidence_score = verification_analysis["confidence"]
            result.supporting_sources = verification_analysis["supporting"]
            result.contradicting_sources = verification_analysis["contradicting"]
            result.verification_details.update(verification_analysis["details"])
            
            # Cache the result
            self.verification_cache[cache_key] = result
            
            self.logger.info(f"Verification complete: {result.verdict} (confidence: {result.confidence_score:.2f})")
            
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            result.verdict = "verification_error"
            result.verification_details["error"] = str(e)
        
        return result
    
    def _extract_claim_information(self, claim: Claim) -> Dict[str, Any]:
        """Extract key information from a claim for verification."""
        info = {
            "entities": [],
            "numbers": [],
            "dates": [],
            "locations": [],
            "claim_type": claim.claim_type
        }
        
        # Extract entities using spaCy if available
        if self.nlp:
            doc = self.nlp(claim.text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE"]:
                    info["entities"].append(ent.text)
                elif ent.label_ in ["DATE", "TIME"]:
                    info["dates"].append(ent.text)
                elif ent.label_ == "GPE":
                    info["locations"].append(ent.text)
        
        # Extract numbers and percentages
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|percent|million|billion|thousand)?\b', claim.text)
        info["numbers"] = numbers
        
        # Extract years
        years = re.findall(r'\b(19|20)\d{2}\b', claim.text)
        info["dates"].extend(years)
        
        return info
    
    def _generate_search_queries(self, claim: Claim, key_info: Dict[str, Any]) -> List[str]:
        """Generate effective search queries for claim verification."""
        queries = []
        
        # Base query: claim text simplified
        base_query = self._simplify_claim_for_search(claim.text)
        queries.append(base_query)
        
        # Entity-focused queries
        if key_info["entities"]:
            for entity in key_info["entities"][:2]:  # Top 2 entities
                entity_query = f"{entity} {base_query}"
                queries.append(entity_query)
        
        # Number/statistics focused queries
        if key_info["numbers"]:
            for number in key_info["numbers"][:2]:
                number_query = f"{number} {base_query}"
                queries.append(number_query)
        
        # Add verification-specific terms
        verification_terms = ["study", "research", "report", "data", "statistics"]
        for term in verification_terms[:2]:
            verification_query = f"{base_query} {term}"
            queries.append(verification_query)
        
        # Limit to reasonable number of queries
        return queries[:5]
    
    def _simplify_claim_for_search(self, claim_text: str) -> str:
        """Simplify claim text for effective search."""
        # Remove opinion indicators
        opinion_patterns = [
            r'\bi think\b', r'\bi believe\b', r'\bin my opinion\b',
            r'\bapparently\b', r'\ballegedly\b', r'\bpresumably\b'
        ]
        
        simplified = claim_text
        for pattern in opinion_patterns:
            simplified = re.sub(pattern, '', simplified, flags=re.IGNORECASE)
        
        # Remove uncertainty indicators
        uncertainty_patterns = [
            r'\bmight\b', r'\bmay\b', r'\bcould\b', r'\bpossibly\b',
            r'\bperhaps\b', r'\blikely\b'
        ]
        
        for pattern in uncertainty_patterns:
            simplified = re.sub(pattern, '', simplified, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        simplified = re.sub(r'\s+', ' ', simplified).strip()
        
        return simplified
    
    async def _search_for_evidence(self, query: str) -> List[Dict[str, Any]]:
        """Search for evidence related to a claim."""
        try:
            # Use DuckDuckGo search
            raw_results = self.search_tool.run(query)
            
            # Parse search results
            sources = self._parse_search_results(raw_results, query)
            
            return sources
            
        except Exception as e:
            self.logger.warning(f"Search failed for query '{query}': {e}")
            return []
    
    def _parse_search_results(self, raw_results: str, query: str) -> List[Dict[str, Any]]:
        """Parse search results into structured format."""
        sources = []
        
        if not raw_results:
            return sources
        
        # Simple parsing - in production, you'd want more robust parsing
        lines = raw_results.split('\n')
        for line in lines[:self.max_search_results]:
            if 'http' in line:
                try:
                    parts = line.split(' - ')
                    if len(parts) >= 2:
                        url = parts[0].strip()
                        title = parts[1].strip()
                        snippet = parts[2] if len(parts) > 2 else ""
                        
                        source = {
                            "url": url,
                            "title": title,
                            "snippet": snippet,
                            "query": query,
                            "search_rank": len(sources) + 1
                        }
                        sources.append(source)
                except Exception as e:
                    self.logger.debug(f"Failed to parse search result: {line}")
        
        return sources
    
    def _deduplicate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate sources based on URL."""
        seen_urls = set()
        unique_sources = []
        
        for source in sources:
            url_normalized = self._normalize_url(source["url"])
            if url_normalized not in seen_urls:
                seen_urls.add(url_normalized)
                unique_sources.append(source)
        
        return unique_sources
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        try:
            parsed = urlparse(url)
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            return normalized.lower()
        except Exception:
            return url.lower()
    
    async def _assess_source_credibility(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assess the credibility of sources."""
        credible_sources = []
        
        for source in sources:
            try:
                url = source["url"]
                domain = urlparse(url).netloc.lower()
                
                # Remove www. prefix
                if domain.startswith('www.'):
                    domain = domain[4:]
                
                # Check against known credible sources
                if domain in self.credible_sources:
                    credibility = self.credible_sources[domain]
                    source["credibility_score"] = credibility["credibility"]
                    source["source_type"] = credibility["type"]
                    source["bias"] = credibility["bias"]
                else:
                    # Calculate credibility based on domain characteristics
                    credibility_score = self._calculate_domain_credibility(domain)
                    source["credibility_score"] = credibility_score
                    source["source_type"] = "unknown"
                    source["bias"] = "unknown"
                
                # Only include sources above minimum credibility threshold
                if source["credibility_score"] >= 0.5:
                    credible_sources.append(source)
                    
            except Exception as e:
                self.logger.debug(f"Failed to assess credibility for {source.get('url', 'unknown')}: {e}")
        
        # Sort by credibility score
        credible_sources.sort(key=lambda x: x["credibility_score"], reverse=True)
        
        return credible_sources[:self.max_search_results]
    
    def _calculate_domain_credibility(self, domain: str) -> float:
        """Calculate credibility score for unknown domains."""
        score = 0.5  # Base score
        
        # High credibility indicators
        if domain.endswith('.gov'):
            score += 0.3
        elif domain.endswith('.edu'):
            score += 0.25
        elif domain.endswith('.org'):
            score += 0.1
        
        # Known high-quality domains
        quality_indicators = [
            'university', 'institute', 'academy', 'college',
            'journal', 'research', 'science', 'medical'
        ]
        
        for indicator in quality_indicators:
            if indicator in domain:
                score += 0.1
                break
        
        # Low credibility indicators
        low_quality_indicators = [
            'blog', 'wordpress', 'blogspot', 'tumblr',
            'medium', 'substack'
        ]
        
        for indicator in low_quality_indicators:
            if indicator in domain:
                score -= 0.2
                break
        
        # Commercial domains
        if domain.endswith('.com'):
            score -= 0.05
        
        return max(0.0, min(1.0, score))
    
    async def _extract_source_contents(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract content from credible sources."""
        source_contents = []
        
        for source in sources:
            try:
                content = await self._extract_content_from_url(source["url"])
                if content:
                    source["content"] = content
                    source["content_length"] = len(content)
                    source_contents.append(source)
                    
            except Exception as e:
                self.logger.warning(f"Failed to extract content from {source['url']}: {e}")
        
        return source_contents
    
    async def _extract_content_from_url(self, url: str) -> Optional[str]:
        """Extract text content from a URL."""
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
                    content = "\n".join(elem.get_text(strip=True) for elem in elements)
                    break
            
            # If no specific content area found, get all text
            if not content:
                content = soup.get_text()
            
            # Clean up the content
            lines = content.split('\n')
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            content = '\n'.join(cleaned_lines)
            
            return content if len(content) > 100 else None
            
        except Exception as e:
            self.logger.debug(f"Content extraction failed for {url}: {e}")
            return None
    
    async def _analyze_claim_against_sources(self, claim: Claim, 
                                           source_contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze claim against source contents to determine verification result."""
        
        if not source_contents:
            return {
                "verdict": "insufficient_evidence",
                "confidence": 0.0,
                "supporting": [],
                "contradicting": [],
                "details": {"reason": "No credible sources found"}
            }
        
        supporting_sources = []
        contradicting_sources = []
        neutral_sources = []
        
        # Compare claim against each source
        for source in source_contents:
            similarity_score = self._calculate_semantic_similarity(
                claim.text, source["content"]
            )
            
            stance = self._determine_source_stance(claim.text, source["content"])
            
            source_analysis = {
                "url": source["url"],
                "title": source.get("title", ""),
                "credibility_score": source["credibility_score"],
                "similarity_score": similarity_score,
                "stance": stance,
                "relevant_excerpts": self._extract_relevant_excerpts(
                    claim.text, source["content"]
                )
            }
            
            if stance == "supporting":
                supporting_sources.append(source_analysis)
            elif stance == "contradicting":
                contradicting_sources.append(source_analysis)
            else:
                neutral_sources.append(source_analysis)
        
        # Determine overall verdict
        verdict, confidence = self._determine_overall_verdict(
            supporting_sources, contradicting_sources, neutral_sources
        )
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "supporting": supporting_sources,
            "contradicting": contradicting_sources,
            "details": {
                "total_sources": len(source_contents),
                "supporting_count": len(supporting_sources),
                "contradicting_count": len(contradicting_sources),
                "neutral_count": len(neutral_sources),
                "avg_credibility": np.mean([s["credibility_score"] for s in source_contents])
            }
        }
    
    def _calculate_semantic_similarity(self, claim: str, content: str) -> float:
        """Calculate semantic similarity between claim and content."""
        if not self.sentence_model:
            # Fallback to simple keyword matching
            return self._calculate_keyword_similarity(claim, content)
        
        try:
            # Encode texts
            claim_embedding = self.sentence_model.encode([claim])
            
            # Split content into chunks for better comparison
            content_chunks = self._split_content_into_chunks(content, max_length=500)
            chunk_embeddings = self.sentence_model.encode(content_chunks)
            
            # Calculate similarity with each chunk and take the maximum
            similarities = cosine_similarity(claim_embedding, chunk_embeddings)[0]
            max_similarity = np.max(similarities) if len(similarities) > 0 else 0.0
            
            return float(max_similarity)
            
        except Exception as e:
            self.logger.warning(f"Semantic similarity calculation failed: {e}")
            return self._calculate_keyword_similarity(claim, content)
    
    def _calculate_keyword_similarity(self, claim: str, content: str) -> float:
        """Fallback keyword-based similarity calculation."""
        claim_words = set(claim.lower().split())
        content_words = set(content.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        claim_words -= stop_words
        content_words -= stop_words
        
        if not claim_words:
            return 0.0
        
        intersection = claim_words.intersection(content_words)
        return len(intersection) / len(claim_words)
    
    def _determine_source_stance(self, claim: str, content: str) -> str:
        """Determine if source supports, contradicts, or is neutral to claim."""
        # Look for explicit agreement/disagreement indicators
        supporting_phrases = [
            'confirms', 'supports', 'validates', 'proves', 'demonstrates',
            'shows that', 'indicates that', 'reveals that', 'found that'
        ]
        
        contradicting_phrases = [
            'contradicts', 'disputes', 'refutes', 'disproves', 'denies',
            'however', 'but', 'nevertheless', 'in contrast', 'on the contrary'
        ]
        
        content_lower = content.lower()
        claim_lower = claim.lower()
        
        # Count supporting and contradicting indicators
        support_score = sum(1 for phrase in supporting_phrases if phrase in content_lower)
        contradict_score = sum(1 for phrase in contradicting_phrases if phrase in content_lower)
        
        # Calculate semantic similarity for additional context
        similarity = self._calculate_semantic_similarity(claim, content)
        
        # Make stance determination
        if support_score > contradict_score and similarity > 0.3:
            return "supporting"
        elif contradict_score > support_score and similarity > 0.3:
            return "contradicting"
        elif similarity > 0.5:
            return "supporting"
        else:
            return "neutral"
    
    def _extract_relevant_excerpts(self, claim: str, content: str, max_excerpts: int = 3) -> List[str]:
        """Extract relevant excerpts from content that relate to the claim."""
        # Split content into sentences
        sentences = content.split('.')
        
        # Score sentences based on relevance to claim
        claim_words = set(claim.lower().split())
        scored_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            sentence_words = set(sentence.lower().split())
            overlap = len(claim_words.intersection(sentence_words))
            
            if overlap > 0:
                score = overlap / len(claim_words)
                scored_sentences.append((sentence, score))
        
        # Sort by score and return top excerpts
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        excerpts = [sentence for sentence, score in scored_sentences[:max_excerpts]]
        
        return excerpts
    
    def _determine_overall_verdict(self, supporting: List[Dict], 
                                 contradicting: List[Dict], 
                                 neutral: List[Dict]) -> Tuple[str, float]:
        """Determine overall verdict and confidence based on source analysis."""
        
        total_sources = len(supporting) + len(contradicting) + len(neutral)
        
        if total_sources < self.min_sources_for_verification:
            return "insufficient_evidence", 0.0
        
        # Calculate weighted scores based on source credibility
        support_weight = sum(s["credibility_score"] for s in supporting)
        contradict_weight = sum(s["credibility_score"] for s in contradicting)
        
        # Determine verdict
        if support_weight > contradict_weight * 1.5:
            verdict = "supported"
            confidence = min(0.95, support_weight / total_sources)
        elif contradict_weight > support_weight * 1.5:
            verdict = "contradicted"
            confidence = min(0.95, contradict_weight / total_sources)
        elif abs(support_weight - contradict_weight) < 0.3:
            verdict = "conflicted"
            confidence = 0.3
        else:
            verdict = "unverified"
            confidence = 0.2
        
        # Adjust confidence based on source quality
        avg_credibility = (support_weight + contradict_weight) / max(1, len(supporting) + len(contradicting))
        confidence *= avg_credibility
        
        return verdict, confidence
    
    def _split_content_into_chunks(self, content: str, max_length: int = 500) -> List[str]:
        """Split content into manageable chunks."""
        sentences = content.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += ". " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _get_cache_key(self, claim: Claim) -> str:
        """Generate cache key for claim verification."""
        key_data = f"{claim.text}_{claim.claim_type}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    # AMP capability handlers
    async def handle_fact_verification(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle fact verification capability requests."""
        claim_text = parameters.get("claim", "")
        claim_type = parameters.get("claim_type", "general")
        source = parameters.get("source")
        
        if not claim_text:
            return {"error": "Claim text is required", "result": None}
        
        claim = Claim(
            text=claim_text,
            source=source,
            claim_type=claim_type
        )
        
        result = await self.verify_claim(claim)
        
        return {
            "result": {
                "claim": claim.text,
                "verdict": result.verdict,
                "confidence_score": result.confidence_score,
                "supporting_sources": result.supporting_sources,
                "contradicting_sources": result.contradicting_sources,
                "verification_details": result.verification_details,
                "timestamp": result.timestamp.isoformat()
            },
            "success": True
        }
    
    async def handle_source_credibility(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle source credibility assessment requests."""
        url = parameters.get("url", "")
        
        if not url:
            return {"error": "URL is required", "credibility": None}
        
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            
            if domain in self.credible_sources:
                credibility_info = self.credible_sources[domain]
                credibility_score = credibility_info["credibility"]
            else:
                credibility_score = self._calculate_domain_credibility(domain)
            
            return {
                "url": url,
                "domain": domain,
                "credibility_score": credibility_score,
                "is_known_source": domain in self.credible_sources,
                "assessment_timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            return {"error": f"Failed to assess credibility: {str(e)}", "credibility": None}
    
    async def start_amp_agent(self, agent_id: str = "fact-checker",
                             endpoint: str = "http://localhost:8000") -> AMPClient:
        """Start the AMP agent."""
        
        self.amp_client = await (
            AMPBuilder(agent_id, "Fact Checker Agent")
            .with_framework("crewai")
            .with_transport(TransportType.HTTP, endpoint)
            .add_capability(
                "fact-verification",
                self.handle_fact_verification,
                "Verify factual claims using multiple credible sources",
                "verification",
                input_schema={
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "claim_type": {
                            "type": "string",
                            "enum": ["general", "statistical", "temporal", "causal"],
                            "default": "general"
                        },
                        "source": {"type": "string"}
                    },
                    "required": ["claim"]
                },
                constraints=CapabilityConstraints(response_time_ms=30000)
            )
            .add_capability(
                "source-credibility",
                self.handle_source_credibility,
                "Assess the credibility and reliability of information sources",
                "assessment",
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
    """Main function for testing the fact checker."""
    logging.basicConfig(level=logging.INFO)
    
    # Create fact checker
    fact_checker = FactChecker()
    
    # Start AMP agent
    client = await fact_checker.start_amp_agent()
    
    try:
        print("Fact Checker started. Testing verification functionality...")
        
        # Test claims
        test_claims = [
            Claim(text="COVID-19 vaccines are 95% effective in preventing severe illness", claim_type="statistical"),
            Claim(text="The Earth's climate has warmed by approximately 1.1°C since pre-industrial times", claim_type="statistical"),
            Claim(text="Artificial intelligence will replace all human jobs by 2030", claim_type="temporal")
        ]
        
        for claim in test_claims:
            print(f"\nVerifying claim: {claim.text}")
            print("=" * 80)
            
            result = await fact_checker.verify_claim(claim)
            
            print(f"Verdict: {result.verdict}")
            print(f"Confidence: {result.confidence_score:.2f}")
            print(f"Supporting sources: {len(result.supporting_sources)}")
            print(f"Contradicting sources: {len(result.contradicting_sources)}")
            
            if result.supporting_sources:
                print("\nTop supporting source:")
                top_source = result.supporting_sources[0]
                print(f"- {top_source.get('title', 'N/A')}")
                print(f"  URL: {top_source.get('url', 'N/A')}")
                print(f"  Credibility: {top_source.get('credibility_score', 0):.2f}")
            
            if result.contradicting_sources:
                print("\nTop contradicting source:")
                top_contra = result.contradicting_sources[0]
                print(f"- {top_contra.get('title', 'N/A')}")
                print(f"  URL: {top_contra.get('url', 'N/A')}")
                print(f"  Credibility: {top_contra.get('credibility_score', 0):.2f}")
        
        print("\nFact Checker is running. Press Ctrl+C to stop.")
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())