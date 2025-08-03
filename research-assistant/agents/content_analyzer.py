"""
Content Analyzer Agent for Research Assistant Network

Handles content analysis, key information extraction, and summarization.
Integrates with CrewAI for coordinated research workflows.
"""

import asyncio
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter
from textstat import flesch_reading_ease, lexicon_count

# NLP and text processing
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# CrewAI imports
from crewai import Agent, Task
from langchain.llms import OpenAI
from langchain.schema import BaseOutputParser

# AMP imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))
from amp_client import AMPClient
from amp_types import Capability, CapabilityConstraints, TransportType
from amp_utils import AMPBuilder


@dataclass
class ContentAnalysis:
    """Represents content analysis results."""
    content: str
    summary: str
    key_points: List[str]
    entities: List[Dict[str, Any]]
    keywords: List[Tuple[str, float]]
    topics: List[str]
    sentiment: Dict[str, float]
    readability_score: float
    word_count: int
    language: str
    bias_indicators: List[str]
    factual_claims: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisRequest:
    """Represents a content analysis request."""
    content: str
    url: Optional[str] = None
    source: Optional[str] = None
    analysis_depth: str = "standard"  # "basic", "standard", "comprehensive"
    focus_areas: List[str] = field(default_factory=list)
    language: str = "en"


class ContentAnalyzer:
    """Content analyzer agent that extracts insights from text content."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.ContentAnalyzer")
        
        # Initialize NLP models
        self._initialize_nlp_models()
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Analysis caching
        self.analysis_cache: Dict[str, ContentAnalysis] = {}
        
        # Configuration
        self.max_summary_length = self.config.get("max_summary_length", 500)
        self.max_key_points = self.config.get("max_key_points", 10)
        self.min_keyword_score = self.config.get("min_keyword_score", 0.1)
        
        # AMP client
        self.amp_client: Optional[AMPClient] = None
        
        # CrewAI agent
        self.crew_agent = self._create_crew_agent()
    
    def _initialize_nlp_models(self):
        """Initialize NLP models and tools."""
        try:
            # Load spaCy model for NER and text processing
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("Loaded spaCy English model")
        except OSError:
            self.logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        try:
            # Initialize sentence transformer for semantic analysis
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Loaded SentenceTransformer model")
        except Exception as e:
            self.logger.warning(f"Failed to load SentenceTransformer: {e}")
            self.sentence_model = None
        
        try:
            # Initialize summarization pipeline
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info("Loaded BART summarization model")
        except Exception as e:
            self.logger.warning(f"Failed to load summarization model: {e}")
            self.summarizer = None
        
        try:
            # Initialize sentiment analyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.logger.info("Initialized VADER sentiment analyzer")
        except Exception as e:
            self.logger.warning(f"Failed to initialize sentiment analyzer: {e}")
            self.sentiment_analyzer = None
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            self.logger.info("Downloaded NLTK data")
        except Exception as e:
            self.logger.warning(f"Failed to download NLTK data: {e}")
    
    def _create_crew_agent(self) -> Agent:
        """Create the CrewAI agent for content analysis."""
        return Agent(
            role="Content Analysis Specialist",
            goal="Extract key insights, entities, and summaries from text content with high accuracy",
            backstory="""You are an expert content analyst with expertise in natural language 
            processing, information extraction, and text summarization. You excel at identifying 
            key concepts, entities, sentiment, and extracting the most important information 
            from complex documents.""",
            tools=[],  # We'll add custom tools
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.1)
        )
    
    async def analyze_content(self, request: AnalysisRequest) -> ContentAnalysis:
        """Perform comprehensive content analysis."""
        # Check cache first
        cache_key = self._get_cache_key(request)
        if cache_key in self.analysis_cache:
            self.logger.info("Returning cached analysis")
            return self.analysis_cache[cache_key]
        
        self.logger.info(f"Starting {request.analysis_depth} analysis of content ({len(request.content)} chars)")
        
        # Initialize analysis result
        analysis = ContentAnalysis(
            content=request.content,
            summary="",
            key_points=[],
            entities=[],
            keywords=[],
            topics=[],
            sentiment={},
            readability_score=0.0,
            word_count=len(request.content.split()),
            language=request.language,
            bias_indicators=[],
            factual_claims=[]
        )
        
        # Perform analysis based on depth
        if request.analysis_depth in ["basic", "standard", "comprehensive"]:
            analysis.summary = await self._generate_summary(request.content)
            analysis.key_points = await self._extract_key_points(request.content)
            analysis.sentiment = self._analyze_sentiment(request.content)
            analysis.readability_score = self._calculate_readability(request.content)
        
        if request.analysis_depth in ["standard", "comprehensive"]:
            analysis.entities = self._extract_entities(request.content)
            analysis.keywords = self._extract_keywords(request.content)
            analysis.topics = await self._identify_topics(request.content)
        
        if request.analysis_depth == "comprehensive":
            analysis.bias_indicators = self._detect_bias_indicators(request.content)
            analysis.factual_claims = self._extract_factual_claims(request.content)
        
        # Add metadata
        analysis.metadata = {
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_depth": request.analysis_depth,
            "source_url": request.url,
            "source_name": request.source,
            "focus_areas": request.focus_areas
        }
        
        # Cache the result
        self.analysis_cache[cache_key] = analysis
        
        self.logger.info(f"Analysis complete: {len(analysis.summary)} char summary, {len(analysis.key_points)} key points")
        return analysis
    
    async def _generate_summary(self, content: str) -> str:
        """Generate a concise summary of the content."""
        if not self.summarizer or len(content.split()) < 50:
            # Fallback to extractive summarization
            return self._extractive_summary(content)
        
        try:
            # Split content into chunks if too long
            max_chunk_length = 1024
            chunks = self._split_text_into_chunks(content, max_chunk_length)
            
            summaries = []
            for chunk in chunks:
                if len(chunk.split()) > 30:  # Only summarize substantial chunks
                    result = self.summarizer(
                        chunk, 
                        max_length=min(150, len(chunk.split()) // 3),
                        min_length=30,
                        do_sample=False
                    )
                    summaries.append(result[0]['summary_text'])
            
            # Combine summaries if multiple chunks
            if len(summaries) > 1:
                combined = " ".join(summaries)
                # Summarize the combined summaries if still too long
                if len(combined.split()) > self.max_summary_length // 5:
                    final_result = self.summarizer(
                        combined,
                        max_length=self.max_summary_length // 5,
                        min_length=50,
                        do_sample=False
                    )
                    return final_result[0]['summary_text']
                return combined
            elif summaries:
                return summaries[0]
            else:
                return self._extractive_summary(content)
                
        except Exception as e:
            self.logger.warning(f"BART summarization failed: {e}, falling back to extractive")
            return self._extractive_summary(content)
    
    def _extractive_summary(self, content: str) -> str:
        """Generate extractive summary by selecting important sentences."""
        sentences = sent_tokenize(content)
        if len(sentences) <= 3:
            return content
        
        # Score sentences based on keyword frequency and position
        word_freq = self._calculate_word_frequencies(content)
        sentence_scores = {}
        
        for i, sentence in enumerate(sentences):
            words = word_tokenize(sentence.lower())
            score = 0
            word_count = 0
            
            for word in words:
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            
            if word_count > 0:
                # Position bonus (earlier sentences get slight boost)
                position_bonus = 1.0 - (i / len(sentences)) * 0.1
                sentence_scores[sentence] = (score / word_count) * position_bonus
        
        # Select top sentences
        num_sentences = min(3, max(1, len(sentences) // 4))
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        
        # Maintain original order
        summary_sentences = []
        for sentence in sentences:
            if any(sentence == sent for sent, _ in top_sentences):
                summary_sentences.append(sentence)
        
        return " ".join(summary_sentences)
    
    async def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from the content."""
        sentences = sent_tokenize(content)
        
        # Use multiple strategies to identify key points
        key_points = set()
        
        # 1. Sentences with important keywords
        important_keywords = [
            'important', 'significant', 'key', 'main', 'primary', 'crucial',
            'essential', 'fundamental', 'major', 'critical', 'notable',
            'conclusion', 'result', 'finding', 'discovery', 'breakthrough'
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in important_keywords):
                key_points.add(sentence.strip())
        
        # 2. Sentences with numerical data or statistics
        for sentence in sentences:
            if re.search(r'\b\d+(?:\.\d+)?%?\b', sentence) and len(sentence.split()) >= 5:
                key_points.add(sentence.strip())
        
        # 3. Sentences that start with common key point indicators
        key_indicators = [
            'the study shows', 'research indicates', 'findings suggest',
            'it was found', 'results show', 'data reveals', 'analysis shows'
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(sentence_lower.startswith(indicator) for indicator in key_indicators):
                key_points.add(sentence.strip())
        
        # 4. Sentences with proper nouns (often indicate specific findings)
        if self.nlp:
            for sentence in sentences:
                doc = self.nlp(sentence)
                if len([ent for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]) >= 2:
                    key_points.add(sentence.strip())
        
        # Limit and rank key points
        key_points_list = list(key_points)[:self.max_key_points * 2]
        
        # Score and rank key points
        scored_points = []
        for point in key_points_list:
            score = self._score_key_point(point, content)
            scored_points.append((point, score))
        
        # Return top-scored key points
        scored_points.sort(key=lambda x: x[1], reverse=True)
        return [point for point, _ in scored_points[:self.max_key_points]]
    
    def _score_key_point(self, point: str, full_content: str) -> float:
        """Score a potential key point based on various factors."""
        score = 0.0
        
        # Length factor (moderate length preferred)
        words = point.split()
        if 8 <= len(words) <= 25:
            score += 1.0
        elif 5 <= len(words) <= 30:
            score += 0.5
        
        # Numerical data bonus
        if re.search(r'\b\d+(?:\.\d+)?%?\b', point):
            score += 0.5
        
        # Important keyword bonus
        important_words = ['significant', 'important', 'key', 'main', 'shows', 'found']
        for word in important_words:
            if word in point.lower():
                score += 0.3
        
        # Sentence structure bonus (complete sentences preferred)
        if point.endswith('.') and point[0].isupper():
            score += 0.2
        
        return score
    
    def _analyze_sentiment(self, content: str) -> Dict[str, float]:
        """Analyze sentiment of the content."""
        if not self.sentiment_analyzer:
            return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(content)
            return {
                "compound": scores['compound'],
                "positive": scores['pos'],
                "negative": scores['neg'],
                "neutral": scores['neu']
            }
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed: {e}")
            return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score using Flesch Reading Ease."""
        try:
            return flesch_reading_ease(content)
        except Exception as e:
            self.logger.warning(f"Readability calculation failed: {e}")
            # Fallback calculation
            sentences = len(sent_tokenize(content))
            words = len(word_tokenize(content))
            if sentences == 0 or words == 0:
                return 0.0
            avg_sentence_length = words / sentences
            return max(0, 100 - avg_sentence_length * 2)
    
    def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract named entities from the content."""
        entities = []
        
        if self.nlp:
            # Use spaCy for entity extraction
            doc = self.nlp(content)
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "description": spacy.explain(ent.label_),
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": getattr(ent, 'score', 1.0)
                })
        else:
            # Fallback to NLTK
            try:
                tokens = word_tokenize(content)
                pos_tags = pos_tag(tokens)
                chunks = ne_chunk(pos_tags)
                
                current_entity = []
                current_label = None
                
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        if current_entity and current_label:
                            entities.append({
                                "text": " ".join(current_entity),
                                "label": current_label,
                                "description": current_label,
                                "confidence": 0.8
                            })
                        current_entity = [token for token, pos in chunk.leaves()]
                        current_label = chunk.label()
                    else:
                        if current_entity and current_label:
                            entities.append({
                                "text": " ".join(current_entity),
                                "label": current_label,
                                "description": current_label,
                                "confidence": 0.8
                            })
                            current_entity = []
                            current_label = None
                
                # Handle last entity
                if current_entity and current_label:
                    entities.append({
                        "text": " ".join(current_entity),
                        "label": current_label,
                        "description": current_label,
                        "confidence": 0.8
                    })
                    
            except Exception as e:
                self.logger.warning(f"NLTK entity extraction failed: {e}")
        
        # Deduplicate and filter entities
        unique_entities = {}
        for entity in entities:
            key = (entity["text"].lower(), entity["label"])
            if key not in unique_entities or unique_entities[key]["confidence"] < entity["confidence"]:
                unique_entities[key] = entity
        
        return list(unique_entities.values())
    
    def _extract_keywords(self, content: str) -> List[Tuple[str, float]]:
        """Extract keywords with TF-IDF-like scoring."""
        # Calculate word frequencies
        word_freq = self._calculate_word_frequencies(content)
        
        # Get total word count
        total_words = sum(word_freq.values())
        
        # Calculate TF scores
        tf_scores = {word: freq / total_words for word, freq in word_freq.items()}
        
        # Simple IDF approximation (in practice, you'd use a corpus)
        # For now, just boost longer words and penalize very common words
        keyword_scores = {}
        for word, tf_score in tf_scores.items():
            # Length bonus
            length_bonus = min(2.0, len(word) / 5.0)
            
            # Frequency penalty for very common words
            freq_penalty = 1.0 if word_freq[word] <= 5 else 0.5
            
            # Final score
            score = tf_score * length_bonus * freq_penalty
            
            if score >= self.min_keyword_score:
                keyword_scores[word] = score
        
        # Sort by score and return top keywords
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:20]  # Top 20 keywords
    
    async def _identify_topics(self, content: str) -> List[str]:
        """Identify main topics in the content."""
        # Simple topic identification based on entity clustering and keyword analysis
        topics = set()
        
        # Extract entities and use them as potential topics
        entities = self._extract_entities(content)
        
        # Group entities by type
        entity_groups = {}
        for entity in entities:
            label = entity["label"]
            if label not in entity_groups:
                entity_groups[label] = []
            entity_groups[label].append(entity["text"])
        
        # Convert entity groups to topics
        for label, entity_list in entity_groups.items():
            if len(entity_list) >= 2:  # Multiple entities of same type suggest a topic
                topics.add(f"{label.lower()}_entities")
        
        # Use keywords to identify additional topics
        keywords = self._extract_keywords(content)
        top_keywords = [word for word, score in keywords[:10]]
        
        # Simple topic clustering based on keyword co-occurrence
        sentences = sent_tokenize(content)
        keyword_cooccurrence = {}
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_keywords = [kw for kw in top_keywords if kw in sentence_lower]
            
            if len(sentence_keywords) >= 2:
                for i, kw1 in enumerate(sentence_keywords):
                    for kw2 in sentence_keywords[i+1:]:
                        pair = tuple(sorted([kw1, kw2]))
                        keyword_cooccurrence[pair] = keyword_cooccurrence.get(pair, 0) + 1
        
        # Add topics based on keyword pairs
        for (kw1, kw2), count in keyword_cooccurrence.items():
            if count >= 2:
                topics.add(f"{kw1}_{kw2}")
        
        return list(topics)[:10]  # Limit to top 10 topics
    
    def _detect_bias_indicators(self, content: str) -> List[str]:
        """Detect potential bias indicators in the content."""
        bias_indicators = []
        
        # Emotional language indicators
        emotional_words = [
            'shocking', 'outrageous', 'devastating', 'incredible', 'amazing',
            'terrible', 'awful', 'fantastic', 'unbelievable', 'ridiculous'
        ]
        
        # Opinion indicators
        opinion_phrases = [
            'i think', 'i believe', 'in my opinion', 'it seems', 'clearly',
            'obviously', 'undoubtedly', 'certainly', 'definitely'
        ]
        
        # Absolute statements
        absolute_words = [
            'always', 'never', 'all', 'none', 'every', 'completely',
            'totally', 'absolutely', 'impossible', 'guaranteed'
        ]
        
        content_lower = content.lower()
        
        # Check for emotional language
        found_emotional = [word for word in emotional_words if word in content_lower]
        if found_emotional:
            bias_indicators.append(f"Emotional language: {', '.join(found_emotional[:3])}")
        
        # Check for opinion phrases
        found_opinions = [phrase for phrase in opinion_phrases if phrase in content_lower]
        if found_opinions:
            bias_indicators.append(f"Opinion indicators: {', '.join(found_opinions[:2])}")
        
        # Check for absolute statements
        found_absolutes = [word for word in absolute_words if word in content_lower]
        if found_absolutes:
            bias_indicators.append(f"Absolute statements: {', '.join(found_absolutes[:3])}")
        
        # Check for lack of sources
        if not re.search(r'\b(study|research|report|survey|according to)\b', content_lower):
            bias_indicators.append("Limited source attribution")
        
        return bias_indicators
    
    def _extract_factual_claims(self, content: str) -> List[str]:
        """Extract factual claims that can be verified."""
        claims = []
        sentences = sent_tokenize(content)
        
        # Patterns that often indicate factual claims
        factual_patterns = [
            r'\b\d+(?:\.\d+)?%\b',  # Percentages
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\s+(?:people|users|customers|dollars|million|billion)\b',  # Numbers with units
            r'\b(?:study|research|report|survey)\s+(?:shows|indicates|finds|reveals)\b',  # Study references
            r'\b(?:according to|data from|statistics show)\b',  # Data references
            r'\b(?:in \d{4}|since \d{4}|by \d{4})\b',  # Date references
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            # Check if sentence matches factual patterns
            matches_pattern = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in factual_patterns)
            
            if matches_pattern:
                claims.append(sentence)
            
            # Also include sentences with specific claim indicators
            claim_indicators = [
                'reported that', 'announced that', 'confirmed that',
                'revealed that', 'discovered that', 'found that'
            ]
            
            if any(indicator in sentence.lower() for indicator in claim_indicators):
                claims.append(sentence)
        
        return claims[:10]  # Limit to top 10 claims
    
    def _calculate_word_frequencies(self, content: str) -> Dict[str, int]:
        """Calculate word frequencies, excluding stop words."""
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            stop_words = set()
        
        words = word_tokenize(content.lower())
        word_freq = {}
        
        for word in words:
            if (word.isalpha() and 
                len(word) > 2 and 
                word not in stop_words):
                word_freq[word] = word_freq.get(word, 0) + 1
        
        return word_freq
    
    def _split_text_into_chunks(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks of maximum length."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _get_cache_key(self, request: AnalysisRequest) -> str:
        """Generate cache key for analysis request."""
        import hashlib
        key_data = f"{request.content[:100]}_{request.analysis_depth}_{sorted(request.focus_areas)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    # AMP capability handlers
    async def handle_content_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle content analysis capability requests."""
        request = AnalysisRequest(
            content=parameters.get("content", ""),
            url=parameters.get("url"),
            source=parameters.get("source"),
            analysis_depth=parameters.get("analysis_depth", "standard"),
            focus_areas=parameters.get("focus_areas", []),
            language=parameters.get("language", "en")
        )
        
        if not request.content:
            return {"error": "Content is required", "analysis": None}
        
        analysis = await self.analyze_content(request)
        
        return {
            "analysis": {
                "summary": analysis.summary,
                "key_points": analysis.key_points,
                "entities": analysis.entities,
                "keywords": [{"word": word, "score": score} for word, score in analysis.keywords],
                "topics": analysis.topics,
                "sentiment": analysis.sentiment,
                "readability_score": analysis.readability_score,
                "word_count": analysis.word_count,
                "language": analysis.language,
                "bias_indicators": analysis.bias_indicators,
                "factual_claims": analysis.factual_claims,
                "metadata": analysis.metadata
            },
            "success": True
        }
    
    async def handle_summarization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text summarization capability requests."""
        content = parameters.get("content", "")
        max_length = parameters.get("max_length", self.max_summary_length)
        
        if not content:
            return {"error": "Content is required", "summary": None}
        
        # Temporarily adjust max length
        original_max = self.max_summary_length
        self.max_summary_length = max_length
        
        try:
            summary = await self._generate_summary(content)
            return {
                "summary": summary,
                "original_length": len(content),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(content) if content else 0,
                "success": True
            }
        finally:
            self.max_summary_length = original_max
    
    async def handle_key_extraction(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle key information extraction capability requests."""
        content = parameters.get("content", "")
        max_points = parameters.get("max_points", self.max_key_points)
        
        if not content:
            return {"error": "Content is required", "key_points": []}
        
        # Temporarily adjust max points
        original_max = self.max_key_points
        self.max_key_points = max_points
        
        try:
            key_points = await self._extract_key_points(content)
            return {
                "key_points": key_points,
                "total_points": len(key_points),
                "success": True
            }
        finally:
            self.max_key_points = original_max
    
    async def start_amp_agent(self, agent_id: str = "content-analyzer",
                             endpoint: str = "http://localhost:8000") -> AMPClient:
        """Start the AMP agent."""
        
        self.amp_client = await (
            AMPBuilder(agent_id, "Content Analyzer Agent")
            .with_framework("crewai")
            .with_transport(TransportType.HTTP, endpoint)
            .add_capability(
                "content-analysis",
                self.handle_content_analysis,
                "Analyze content and extract insights, entities, and summaries",
                "analysis",
                input_schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "url": {"type": "string"},
                        "source": {"type": "string"},
                        "analysis_depth": {
                            "type": "string", 
                            "enum": ["basic", "standard", "comprehensive"],
                            "default": "standard"
                        },
                        "focus_areas": {"type": "array", "items": {"type": "string"}},
                        "language": {"type": "string", "default": "en"}
                    },
                    "required": ["content"]
                },
                constraints=CapabilityConstraints(response_time_ms=15000)
            )
            .add_capability(
                "text-summarization",
                self.handle_summarization,
                "Generate concise summaries of text content",
                "generation",
                input_schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "max_length": {"type": "integer", "default": 500}
                    },
                    "required": ["content"]
                }
            )
            .add_capability(
                "key-extraction",
                self.handle_key_extraction,
                "Extract key points and important information",
                "extraction",
                input_schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "max_points": {"type": "integer", "default": 10}
                    },
                    "required": ["content"]
                }
            )
            .build()
        )
        
        return self.amp_client


async def main():
    """Main function for testing the content analyzer."""
    logging.basicConfig(level=logging.INFO)
    
    # Create content analyzer
    analyzer = ContentAnalyzer()
    
    # Start AMP agent
    client = await analyzer.start_amp_agent()
    
    try:
        print("Content Analyzer started. Testing analysis functionality...")
        
        # Test content
        test_content = """
        A recent study published in Nature Communications reveals that artificial intelligence 
        systems are becoming increasingly sophisticated in their ability to analyze complex 
        datasets. The research, conducted by a team of scientists at MIT and Stanford University, 
        found that machine learning algorithms can now identify patterns in data that were 
        previously undetectable by traditional statistical methods.
        
        The study analyzed over 1 million data points across various domains including healthcare, 
        finance, and climate science. Results showed a 45% improvement in pattern recognition 
        accuracy compared to conventional approaches. Dr. Sarah Johnson, lead researcher, stated 
        that "these findings represent a significant breakthrough in our understanding of how AI 
        can augment human analytical capabilities."
        
        The implications of this research are far-reaching, potentially revolutionizing fields 
        from medical diagnosis to financial fraud detection. However, researchers caution that 
        ethical considerations must be carefully addressed as these technologies become more 
        widespread.
        """
        
        # Test comprehensive analysis
        request = AnalysisRequest(
            content=test_content,
            analysis_depth="comprehensive",
            focus_areas=["research", "technology", "ethics"]
        )
        
        analysis = await analyzer.analyze_content(request)
        
        print(f"\nContent Analysis Results:")
        print("=" * 60)
        print(f"Word Count: {analysis.word_count}")
        print(f"Readability Score: {analysis.readability_score:.1f}")
        print(f"Sentiment: {analysis.sentiment}")
        print(f"\nSummary:\n{analysis.summary}")
        print(f"\nKey Points:")
        for i, point in enumerate(analysis.key_points, 1):
            print(f"{i}. {point}")
        
        print(f"\nEntities:")
        for entity in analysis.entities[:5]:  # Show top 5
            print(f"- {entity['text']} ({entity['label']})")
        
        print(f"\nTop Keywords:")
        for word, score in analysis.keywords[:10]:
            print(f"- {word}: {score:.3f}")
        
        print(f"\nTopics: {', '.join(analysis.topics)}")
        
        if analysis.bias_indicators:
            print(f"\nBias Indicators: {', '.join(analysis.bias_indicators)}")
        
        if analysis.factual_claims:
            print(f"\nFactual Claims:")
            for claim in analysis.factual_claims[:3]:
                print(f"- {claim}")
        
        print("\nContent Analyzer is running. Press Ctrl+C to stop.")
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())