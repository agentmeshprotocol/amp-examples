"""
Content Extraction Tools for Research Assistant Network

Specialized tools for content extraction, analysis, and processing.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re
import json

# NLP libraries
import spacy
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Text processing
from textstat import flesch_reading_ease, flesch_kincaid_grade
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


@dataclass
class ExtractedContent:
    """Structured extracted content."""
    text: str
    title: Optional[str] = None
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    word_count: int = 0
    reading_time_minutes: int = 0
    language: str = "en"
    content_type: str = "article"
    metadata: Dict[str, Any] = None


@dataclass
class ContentSummary:
    """Content summary with key information."""
    original_text: str
    summary: str
    key_points: List[str]
    entities: List[Dict[str, Any]]
    keywords: List[Tuple[str, float]]
    sentiment: Dict[str, float]
    readability_score: float
    topics: List[str]


class ContentExtractionTools:
    """Advanced content extraction and analysis tools."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.ContentExtractionTools")
        
        # Initialize NLP models
        self._initialize_nlp()
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Configuration
        self.min_content_length = self.config.get("min_content_length", 100)
        self.max_summary_ratio = self.config.get("max_summary_ratio", 0.3)
        self.reading_speed_wpm = self.config.get("reading_speed_wpm", 200)
    
    def _initialize_nlp(self):
        """Initialize NLP models and tools."""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("Loaded spaCy English model")
        except OSError:
            self.logger.warning("spaCy English model not found")
            self.nlp = None
        
        try:
            # Initialize summarization pipeline
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1  # Use CPU
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
        except Exception as e:
            self.logger.warning(f"Failed to download NLTK data: {e}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Remove repeated punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Fix common encoding issues
        text = text.replace('\u2019', "'")  # Right single quotation mark
        text = text.replace('\u2018', "'")  # Left single quotation mark
        text = text.replace('\u201c', '"')  # Left double quotation mark
        text = text.replace('\u201d', '"')  # Right double quotation mark
        text = text.replace('\u2013', '-')  # En dash
        text = text.replace('\u2014', '--') # Em dash
        
        return text.strip()
    
    def extract_structure(self, text: str) -> Dict[str, Any]:
        """Extract structural elements from text."""
        structure = {
            "paragraphs": [],
            "sentences": [],
            "headings": [],
            "lists": [],
            "quotes": []
        }
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        structure["paragraphs"] = paragraphs
        
        # Extract sentences
        try:
            sentences = sent_tokenize(text)
            structure["sentences"] = sentences
        except Exception:
            structure["sentences"] = text.split('.')
        
        # Identify potential headings (short lines, often capitalized)
        for para in paragraphs:
            if len(para) < 100 and para.isupper():
                structure["headings"].append(para)
        
        # Extract quotes (text in quotation marks)
        quotes = re.findall(r'"([^"]*)"', text)
        structure["quotes"] = quotes[:10]  # Limit to 10 quotes
        
        # Identify lists (lines starting with bullets or numbers)
        list_items = re.findall(r'(?:^|\n)[\s]*[•\-\*\d+\.]\s*(.+)', text, re.MULTILINE)
        structure["lists"] = list_items[:20]  # Limit to 20 items
        
        return structure
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        entities = []
        
        if self.nlp:
            try:
                doc = self.nlp(text[:1000000])  # Limit text length for processing
                
                for ent in doc.ents:
                    entity = {
                        "text": ent.text,
                        "label": ent.label_,
                        "description": spacy.explain(ent.label_),
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": getattr(ent, 'score', 1.0)
                    }
                    entities.append(entity)
                
                # Deduplicate entities
                unique_entities = {}
                for entity in entities:
                    key = (entity["text"].lower(), entity["label"])
                    if key not in unique_entities or unique_entities[key]["confidence"] < entity["confidence"]:
                        unique_entities[key] = entity
                
                return list(unique_entities.values())
                
            except Exception as e:
                self.logger.warning(f"Entity extraction failed: {e}")
        
        # Fallback: simple regex-based entity extraction
        return self._extract_entities_regex(text)
    
    def _extract_entities_regex(self, text: str) -> List[Dict[str, Any]]:
        """Fallback entity extraction using regex patterns."""
        entities = []
        
        # Dates
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "text": match.group(),
                    "label": "DATE",
                    "description": "Date",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.8
                })
        
        # URLs and emails
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        for pattern, label in [(url_pattern, "URL"), (email_pattern, "EMAIL")]:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    "text": match.group(),
                    "label": label,
                    "description": label.title(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9
                })
        
        return entities[:50]  # Limit results
    
    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF."""
        try:
            # Clean text and split into sentences
            sentences = sent_tokenize(text)
            
            if len(sentences) < 2:
                return []
            
            # Initialize TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get mean TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Create keyword-score pairs
            keywords = list(zip(feature_names, mean_scores))
            
            # Sort by score and return top keywords
            keywords.sort(key=lambda x: x[1], reverse=True)
            
            return keywords[:max_keywords]
            
        except Exception as e:
            self.logger.warning(f"Keyword extraction failed: {e}")
            return self._extract_keywords_simple(text, max_keywords)
    
    def _extract_keywords_simple(self, text: str, max_keywords: int) -> List[Tuple[str, float]]:
        """Simple keyword extraction based on word frequency."""
        try:
            # Get stop words
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
        
        # Tokenize and filter
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and len(word) > 2 and word not in stop_words]
        
        # Count frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Calculate TF scores
        total_words = len(words)
        tf_scores = {word: freq / total_words for word, freq in word_freq.items()}
        
        # Sort and return
        keywords = sorted(tf_scores.items(), key=lambda x: x[1], reverse=True)
        return keywords[:max_keywords]
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text."""
        if self.sentiment_analyzer:
            try:
                scores = self.sentiment_analyzer.polarity_scores(text)
                return {
                    "compound": scores['compound'],
                    "positive": scores['pos'],
                    "negative": scores['neg'],
                    "neutral": scores['neu']
                }
            except Exception as e:
                self.logger.warning(f"Sentiment analysis failed: {e}")
        
        # Fallback: simple sentiment analysis
        return self._analyze_sentiment_simple(text)
    
    def _analyze_sentiment_simple(self, text: str) -> Dict[str, float]:
        """Simple sentiment analysis based on keyword counts."""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'positive', 'success', 'improve']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'negative', 'problem', 'issue', 'fail']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        positive_ratio = positive_count / total_sentiment_words
        negative_ratio = negative_count / total_sentiment_words
        
        compound = positive_ratio - negative_ratio
        
        return {
            "compound": compound,
            "positive": positive_ratio,
            "negative": negative_ratio,
            "neutral": 1.0 - positive_ratio - negative_ratio
        }
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate various readability metrics."""
        try:
            flesch_ease = flesch_reading_ease(text)
            flesch_grade = flesch_kincaid_grade(text)
            
            # Calculate additional metrics
            sentences = len(sent_tokenize(text))
            words = len(word_tokenize(text))
            avg_sentence_length = words / sentences if sentences > 0 else 0
            
            return {
                "flesch_reading_ease": flesch_ease,
                "flesch_kincaid_grade": flesch_grade,
                "avg_sentence_length": avg_sentence_length,
                "reading_level": self._interpret_flesch_score(flesch_ease)
            }
            
        except Exception as e:
            self.logger.warning(f"Readability calculation failed: {e}")
            return {
                "flesch_reading_ease": 50.0,
                "flesch_kincaid_grade": 10.0,
                "avg_sentence_length": 20.0,
                "reading_level": "average"
            }
    
    def _interpret_flesch_score(self, score: float) -> str:
        """Interpret Flesch reading ease score."""
        if score >= 90:
            return "very_easy"
        elif score >= 80:
            return "easy"
        elif score >= 70:
            return "fairly_easy"
        elif score >= 60:
            return "standard"
        elif score >= 50:
            return "fairly_difficult"
        elif score >= 30:
            return "difficult"
        else:
            return "very_difficult"
    
    def generate_summary(self, text: str, max_length: int = 150) -> str:
        """Generate a summary of the text."""
        if self.summarizer and len(text) > 100:
            try:
                # Split text if too long
                if len(text) > 1024:
                    chunks = self._split_text_chunks(text, 1024)
                    summaries = []
                    
                    for chunk in chunks[:3]:  # Limit to 3 chunks
                        if len(chunk.split()) > 30:
                            result = self.summarizer(
                                chunk,
                                max_length=max_length // len(chunks),
                                min_length=20,
                                do_sample=False
                            )
                            summaries.append(result[0]['summary_text'])
                    
                    return ' '.join(summaries)
                else:
                    result = self.summarizer(
                        text,
                        max_length=max_length,
                        min_length=30,
                        do_sample=False
                    )
                    return result[0]['summary_text']
                    
            except Exception as e:
                self.logger.warning(f"AI summarization failed: {e}")
        
        # Fallback: extractive summarization
        return self._generate_extractive_summary(text, max_length)
    
    def _generate_extractive_summary(self, text: str, max_length: int) -> str:
        """Generate extractive summary by selecting important sentences."""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 2:
            return text
        
        # Score sentences
        sentence_scores = {}
        keywords = self.extract_keywords(text, 10)
        keyword_dict = {word: score for word, score in keywords}
        
        for sentence in sentences:
            score = 0
            words = word_tokenize(sentence.lower())
            
            # Score based on keyword presence
            for word in words:
                if word in keyword_dict:
                    score += keyword_dict[word]
            
            # Position bonus (earlier sentences often more important)
            position_bonus = 1.0 - (sentences.index(sentence) / len(sentences)) * 0.3
            score *= position_bonus
            
            sentence_scores[sentence] = score
        
        # Select top sentences
        num_sentences = max(1, min(3, len(sentences) // 3))
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        
        # Maintain original order
        summary_sentences = []
        for sentence in sentences:
            if any(sentence == sent for sent, _ in top_sentences):
                summary_sentences.append(sentence)
        
        summary = ' '.join(summary_sentences)
        
        # Truncate if too long
        if len(summary.split()) > max_length:
            words = summary.split()[:max_length]
            summary = ' '.join(words)
            if not summary.endswith('.'):
                summary += '...'
        
        return summary
    
    def _split_text_chunks(self, text: str, max_length: int) -> List[str]:
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
    
    def extract_facts_and_claims(self, text: str) -> List[str]:
        """Extract factual claims and statements from text."""
        claims = []
        sentences = sent_tokenize(text)
        
        # Patterns that often indicate factual claims
        fact_patterns = [
            r'\b\d+(?:\.\d+)?%\b',  # Percentages
            r'\b\d+(?:,\d{3})*\s+(?:people|users|dollars|million|billion)\b',  # Numbers with units
            r'\b(?:study|research|data)\s+(?:shows|indicates|reveals)\b',  # Study references
            r'\b(?:according to|statistics show)\b',  # Data references
        ]
        
        for sentence in sentences:
            # Check for factual patterns
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in fact_patterns):
                claims.append(sentence.strip())
            
            # Check for definitive statements
            definitive_indicators = [
                'is', 'are', 'was', 'were', 'has', 'have', 'will', 'can', 'cannot'
            ]
            
            if (any(indicator in sentence.lower() for indicator in definitive_indicators) and
                len(sentence.split()) >= 5 and
                not sentence.strip().endswith('?')):
                claims.append(sentence.strip())
        
        return claims[:10]  # Limit to 10 claims
    
    def calculate_content_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate comprehensive content metrics."""
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        metrics = {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
            "character_count": len(text),
            "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
            "reading_time_minutes": len(words) / self.reading_speed_wpm,
            "unique_words": len(set(word.lower() for word in words if word.isalpha())),
            "lexical_diversity": len(set(word.lower() for word in words if word.isalpha())) / len(words) if words else 0
        }
        
        return metrics
    
    async def comprehensive_analysis(self, text: str) -> ContentSummary:
        """Perform comprehensive content analysis."""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Generate summary
        summary = self.generate_summary(cleaned_text)
        
        # Extract key information
        entities = self.extract_entities(cleaned_text)
        keywords = self.extract_keywords(cleaned_text)
        sentiment = self.analyze_sentiment(cleaned_text)
        readability = self.calculate_readability(cleaned_text)
        
        # Extract key points (important sentences)
        sentences = sent_tokenize(cleaned_text)
        scored_sentences = []
        
        for sentence in sentences[:20]:  # Limit to first 20 sentences
            score = 0
            
            # Score based on length
            words = sentence.split()
            if 8 <= len(words) <= 25:
                score += 1
            
            # Score based on keywords
            for keyword, keyword_score in keywords[:10]:
                if keyword.lower() in sentence.lower():
                    score += keyword_score
            
            # Score based on position
            position_score = 1.0 - (sentences.index(sentence) / len(sentences)) * 0.5
            score *= position_score
            
            scored_sentences.append((sentence, score))
        
        # Select top sentences as key points
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        key_points = [sentence for sentence, _ in scored_sentences[:5]]
        
        # Identify topics based on entities and keywords
        topics = []
        
        # Add entity-based topics
        entity_types = {}
        for entity in entities:
            entity_type = entity['label']
            if entity_type not in entity_types:
                entity_types[entity_type] = 0
            entity_types[entity_type] += 1
        
        for entity_type, count in entity_types.items():
            if count >= 2:
                topics.append(entity_type.lower().replace('_', ' '))
        
        # Add keyword-based topics
        for keyword, score in keywords[:5]:
            if score > 0.1 and len(keyword.split()) <= 2:
                topics.append(keyword)
        
        return ContentSummary(
            original_text=text,
            summary=summary,
            key_points=key_points,
            entities=entities,
            keywords=keywords,
            sentiment=sentiment,
            readability_score=readability.get('flesch_reading_ease', 50.0),
            topics=topics[:10]
        )


# Utility functions
def extract_text_from_html(html_content: str) -> str:
    """Extract clean text from HTML content."""
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove unwanted elements
    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        element.decompose()
    
    # Extract text
    text = soup.get_text()
    
    # Clean up
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    
    return '\n'.join(cleaned_lines)


def detect_language(text: str) -> str:
    """Simple language detection."""
    # This is a very basic implementation
    # In production, use a proper language detection library
    
    english_indicators = ['the', 'and', 'of', 'to', 'a', 'in', 'for', 'is', 'on', 'that']
    spanish_indicators = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no']
    french_indicators = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir']
    
    text_lower = text.lower()
    
    english_count = sum(1 for word in english_indicators if word in text_lower)
    spanish_count = sum(1 for word in spanish_indicators if word in text_lower)
    french_count = sum(1 for word in french_indicators if word in text_lower)
    
    if english_count >= spanish_count and english_count >= french_count:
        return "en"
    elif spanish_count >= french_count:
        return "es"
    elif french_count > 0:
        return "fr"
    else:
        return "unknown"


if __name__ == "__main__":
    async def test_content_extraction():
        """Test content extraction tools."""
        logging.basicConfig(level=logging.INFO)
        
        tools = ContentExtractionTools()
        
        # Test text
        test_text = """
        Artificial Intelligence in Healthcare: A Revolutionary Transformation
        
        The integration of artificial intelligence (AI) in healthcare has emerged as one of 
        the most significant technological advances of the 21st century. Recent studies 
        indicate that AI applications in medical diagnosis have achieved accuracy rates 
        exceeding 95% in certain specialties.
        
        Dr. Sarah Johnson, a leading researcher at Stanford Medical Center, states that 
        "AI is fundamentally changing how we approach patient care and medical research." 
        The technology has shown particular promise in areas such as radiology, pathology, 
        and drug discovery.
        
        Key benefits include:
        • Improved diagnostic accuracy
        • Reduced treatment costs by 30-40%
        • Enhanced patient outcomes
        • Accelerated drug development processes
        
        However, challenges remain, including data privacy concerns and the need for 
        regulatory frameworks. The FDA has approved over 100 AI-based medical devices 
        since 2018, demonstrating the growing acceptance of these technologies.
        """
        
        print("Testing comprehensive content analysis...")
        analysis = await tools.comprehensive_analysis(test_text)
        
        print(f"\\nSummary:\\n{analysis.summary}")
        print(f"\\nKey Points:")
        for i, point in enumerate(analysis.key_points, 1):
            print(f"{i}. {point}")
        
        print(f"\\nEntities:")
        for entity in analysis.entities[:5]:
            print(f"- {entity['text']} ({entity['label']})")
        
        print(f"\\nTop Keywords:")
        for keyword, score in analysis.keywords[:5]:
            print(f"- {keyword}: {score:.3f}")
        
        print(f"\\nSentiment: {analysis.sentiment}")
        print(f"Readability Score: {analysis.readability_score:.1f}")
        print(f"Topics: {', '.join(analysis.topics)}")
    
    asyncio.run(test_content_extraction())