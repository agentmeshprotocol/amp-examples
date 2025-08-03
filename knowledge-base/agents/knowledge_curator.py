"""
Knowledge Curator Agent

This agent manages knowledge quality, content validation, deduplication,
and provides analytics and insights about the knowledge base.
"""

import asyncio
import json
import logging
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_reading_ease, flesch_kincaid_grade

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared-lib'))
from amp_client import AMPClient
from amp_types import AMPMessage, MessageType

logger = logging.getLogger(__name__)


class QualityAnalyzer:
    """Analyzes content quality and provides quality scores"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Quality metrics weights
        self.quality_weights = {
            'readability': 0.2,
            'coherence': 0.25,
            'completeness': 0.2,
            'accuracy_indicators': 0.15,
            'relevance': 0.1,
            'freshness': 0.1
        }
    
    async def analyze_content_quality(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall content quality"""
        text = content.get('text', '')
        metadata = content.get('metadata', {})
        
        if not text:
            return {'quality_score': 0.0, 'issues': ['No text content']}
        
        # Run quality checks
        readability = await self._analyze_readability(text)
        coherence = await self._analyze_coherence(text)
        completeness = await self._analyze_completeness(text, metadata)
        accuracy = await self._analyze_accuracy_indicators(text)
        relevance = await self._analyze_relevance(text, metadata)
        freshness = await self._analyze_freshness(metadata)
        
        # Calculate weighted quality score
        quality_score = (
            readability['score'] * self.quality_weights['readability'] +
            coherence['score'] * self.quality_weights['coherence'] +
            completeness['score'] * self.quality_weights['completeness'] +
            accuracy['score'] * self.quality_weights['accuracy_indicators'] +
            relevance['score'] * self.quality_weights['relevance'] +
            freshness['score'] * self.quality_weights['freshness']
        )
        
        # Collect issues
        issues = []
        for analysis in [readability, coherence, completeness, accuracy, relevance, freshness]:
            issues.extend(analysis.get('issues', []))
        
        return {
            'quality_score': quality_score,
            'detailed_scores': {
                'readability': readability,
                'coherence': coherence,
                'completeness': completeness,
                'accuracy_indicators': accuracy,
                'relevance': relevance,
                'freshness': freshness
            },
            'issues': issues,
            'recommendations': await self._generate_recommendations(quality_score, issues)
        }
    
    async def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze text readability"""
        try:
            flesch_score = flesch_reading_ease(text)
            grade_level = flesch_kincaid_grade(text)
            
            # Convert Flesch score to 0-1 scale
            # Flesch scores: 90-100 (very easy), 80-90 (easy), 70-80 (fairly easy), etc.
            normalized_score = min(1.0, max(0.0, flesch_score / 100.0))
            
            issues = []
            if flesch_score < 30:
                issues.append("Text is very difficult to read")
            elif flesch_score < 50:
                issues.append("Text is difficult to read")
            
            if grade_level > 16:
                issues.append("Text requires graduate-level education to understand")
            
            return {
                'score': normalized_score,
                'flesch_score': flesch_score,
                'grade_level': grade_level,
                'issues': issues
            }
        except Exception as e:
            logger.error(f"Error analyzing readability: {e}")
            return {'score': 0.5, 'issues': ['Could not analyze readability']}
    
    async def _analyze_coherence(self, text: str) -> Dict[str, Any]:
        """Analyze text coherence and structure"""
        try:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            
            if len(sentences) < 2:
                return {'score': 0.5, 'issues': ['Too few sentences to analyze coherence']}
            
            # Calculate sentence similarity for coherence
            sentence_embeddings = self.model.encode(sentences)
            similarities = []
            
            for i in range(len(sentences) - 1):
                sim = cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[i + 1]])[0][0]
                similarities.append(sim)
            
            avg_coherence = np.mean(similarities)
            
            # Check for coherence issues
            issues = []
            if avg_coherence < 0.3:
                issues.append("Low coherence between sentences")
            
            # Check for repetitive content
            if len(set(sentences)) < len(sentences) * 0.8:
                issues.append("Contains repetitive sentences")
            
            # Check for proper sentence structure
            incomplete_sentences = sum(1 for sent in sentences if len(sent.split()) < 5)
            if incomplete_sentences > len(sentences) * 0.3:
                issues.append("Many incomplete or very short sentences")
            
            return {
                'score': min(1.0, avg_coherence * 2),  # Scale to 0-1
                'average_coherence': avg_coherence,
                'sentence_count': len(sentences),
                'issues': issues
            }
        except Exception as e:
            logger.error(f"Error analyzing coherence: {e}")
            return {'score': 0.5, 'issues': ['Could not analyze coherence']}
    
    async def _analyze_completeness(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content completeness"""
        try:
            doc = self.nlp(text)
            
            # Check for incomplete elements
            issues = []
            score = 1.0
            
            # Check for unfinished sentences
            sentences = [sent.text.strip() for sent in doc.sents]
            for sentence in sentences:
                if sentence.endswith('...') or 'TODO' in sentence.upper():
                    issues.append("Contains incomplete sentences or TODOs")
                    score -= 0.1
                    break
            
            # Check for metadata completeness
            expected_metadata = ['title', 'author', 'date', 'source']
            missing_metadata = [field for field in expected_metadata if field not in metadata]
            if missing_metadata:
                issues.append(f"Missing metadata: {', '.join(missing_metadata)}")
                score -= 0.1 * len(missing_metadata)
            
            # Check content length
            word_count = len(text.split())
            if word_count < 50:
                issues.append("Content is very short")
                score -= 0.2
            elif word_count < 100:
                issues.append("Content is quite short")
                score -= 0.1
            
            return {
                'score': max(0.0, score),
                'word_count': word_count,
                'missing_metadata': missing_metadata,
                'issues': issues
            }
        except Exception as e:
            logger.error(f"Error analyzing completeness: {e}")
            return {'score': 0.5, 'issues': ['Could not analyze completeness']}
    
    async def _analyze_accuracy_indicators(self, text: str) -> Dict[str, Any]:
        """Analyze indicators of content accuracy"""
        try:
            doc = self.nlp(text)
            
            score = 0.8  # Start with good score
            issues = []
            
            # Check for uncertainty markers
            uncertainty_markers = ['maybe', 'probably', 'might', 'could be', 'perhaps', 'possibly']
            uncertainty_count = sum(1 for token in doc if token.text.lower() in uncertainty_markers)
            
            if uncertainty_count > len(doc) * 0.05:  # More than 5% uncertainty markers
                issues.append("High level of uncertainty in content")
                score -= 0.2
            
            # Check for contradictory statements
            contradiction_words = ['however', 'but', 'although', 'contradicts', 'disagrees']
            contradiction_count = sum(1 for token in doc if token.text.lower() in contradiction_words)
            
            if contradiction_count > 3:
                issues.append("May contain contradictory statements")
                score -= 0.1
            
            # Check for factual claims with dates
            date_entities = [ent for ent in doc.ents if ent.label_ == 'DATE']
            if date_entities:
                score += 0.1  # Bonus for dated information
            
            # Check for sources and citations
            citation_indicators = ['according to', 'source:', 'ref:', 'citation', 'study shows']
            has_citations = any(indicator in text.lower() for indicator in citation_indicators)
            
            if has_citations:
                score += 0.1
            else:
                issues.append("No clear sources or citations")
                score -= 0.1
            
            return {
                'score': max(0.0, min(1.0, score)),
                'uncertainty_markers': uncertainty_count,
                'has_citations': has_citations,
                'issues': issues
            }
        except Exception as e:
            logger.error(f"Error analyzing accuracy indicators: {e}")
            return {'score': 0.5, 'issues': ['Could not analyze accuracy indicators']}
    
    async def _analyze_relevance(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content relevance"""
        try:
            # This is a simplified relevance check
            # In a real system, this would compare against domain-specific criteria
            
            doc = self.nlp(text)
            
            # Check for domain-specific entities
            relevant_entities = [ent for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT']]
            
            score = 0.7  # Base relevance score
            issues = []
            
            # Bonus for having relevant entities
            if relevant_entities:
                score += 0.2
            
            # Check if content matches declared topic (if available)
            declared_topic = metadata.get('topic') or metadata.get('category')
            if declared_topic:
                # Simple keyword matching
                if declared_topic.lower() in text.lower():
                    score += 0.1
                else:
                    issues.append("Content may not match declared topic")
                    score -= 0.2
            
            return {
                'score': max(0.0, min(1.0, score)),
                'relevant_entities': len(relevant_entities),
                'declared_topic': declared_topic,
                'issues': issues
            }
        except Exception as e:
            logger.error(f"Error analyzing relevance: {e}")
            return {'score': 0.5, 'issues': ['Could not analyze relevance']}
    
    async def _analyze_freshness(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content freshness"""
        try:
            created_date = metadata.get('created_at') or metadata.get('date')
            processed_date = metadata.get('processed_at')
            
            score = 0.5  # Default score
            issues = []
            
            if created_date:
                try:
                    if isinstance(created_date, str):
                        created_dt = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                    else:
                        created_dt = created_date
                    
                    age_days = (datetime.utcnow() - created_dt.replace(tzinfo=None)).days
                    
                    # Fresher content gets higher scores
                    if age_days < 30:
                        score = 1.0
                    elif age_days < 90:
                        score = 0.8
                    elif age_days < 365:
                        score = 0.6
                    elif age_days < 1825:  # 5 years
                        score = 0.4
                    else:
                        score = 0.2
                        issues.append("Content is very old")
                    
                except Exception:
                    issues.append("Could not parse creation date")
            else:
                issues.append("No creation date available")
            
            return {
                'score': score,
                'age_days': age_days if 'age_days' in locals() else None,
                'created_date': created_date,
                'issues': issues
            }
        except Exception as e:
            logger.error(f"Error analyzing freshness: {e}")
            return {'score': 0.5, 'issues': ['Could not analyze freshness']}
    
    async def _generate_recommendations(self, quality_score: float, issues: List[str]) -> List[str]:
        """Generate recommendations based on quality analysis"""
        recommendations = []
        
        if quality_score < 0.3:
            recommendations.append("Content quality is very low - consider manual review")
        elif quality_score < 0.5:
            recommendations.append("Content quality is below average - improvements needed")
        elif quality_score < 0.7:
            recommendations.append("Content quality is acceptable but can be improved")
        
        # Specific recommendations based on issues
        if any("difficult to read" in issue for issue in issues):
            recommendations.append("Simplify language and sentence structure")
        
        if any("coherence" in issue for issue in issues):
            recommendations.append("Improve logical flow between sentences")
        
        if any("incomplete" in issue for issue in issues):
            recommendations.append("Complete missing content and metadata")
        
        if any("citations" in issue for issue in issues):
            recommendations.append("Add sources and citations for factual claims")
        
        if any("old" in issue for issue in issues):
            recommendations.append("Update content with recent information")
        
        return recommendations


class DuplicationDetector:
    """Detects duplicate and near-duplicate content"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    
    async def find_duplicates(self, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find duplicate and near-duplicate content"""
        if len(content_list) < 2:
            return []
        
        # Extract texts and generate embeddings
        texts = [item.get('text', '') for item in content_list]
        embeddings = self.model.encode(texts)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        duplicates = []
        processed_pairs = set()
        
        for i in range(len(content_list)):
            for j in range(i + 1, len(content_list)):
                if (i, j) in processed_pairs:
                    continue
                
                similarity = similarity_matrix[i][j]
                
                if similarity >= self.similarity_threshold:
                    duplicate_group = {
                        'similarity': similarity,
                        'items': [
                            {
                                'id': content_list[i].get('id'),
                                'text_preview': texts[i][:200] + '...' if len(texts[i]) > 200 else texts[i],
                                'metadata': content_list[i].get('metadata', {})
                            },
                            {
                                'id': content_list[j].get('id'),
                                'text_preview': texts[j][:200] + '...' if len(texts[j]) > 200 else texts[j],
                                'metadata': content_list[j].get('metadata', {})
                            }
                        ],
                        'recommendation': await self._recommend_duplicate_action(
                            content_list[i], content_list[j], similarity
                        )
                    }
                    duplicates.append(duplicate_group)
                    processed_pairs.add((i, j))
        
        return duplicates
    
    async def _recommend_duplicate_action(self, item1: Dict[str, Any], 
                                        item2: Dict[str, Any], similarity: float) -> str:
        """Recommend action for duplicate content"""
        if similarity > 0.95:
            return "Remove one item - content is nearly identical"
        elif similarity > 0.9:
            return "Merge items or keep the higher quality version"
        else:
            return "Review for potential consolidation"


class KnowledgeAnalytics:
    """Provides analytics and insights about the knowledge base"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    async def generate_knowledge_base_report(self, content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive knowledge base analytics report"""
        if not content_list:
            return {'error': 'No content to analyze'}
        
        # Basic statistics
        total_items = len(content_list)
        total_words = sum(len(item.get('text', '').split()) for item in content_list)
        
        # Content type analysis
        content_types = Counter()
        for item in content_list:
            content_type = item.get('metadata', {}).get('mime_type', 'unknown')
            content_types[content_type] += 1
        
        # Quality distribution
        quality_scores = [item.get('quality_score', 0.5) for item in content_list if 'quality_score' in item]
        quality_stats = {
            'average': statistics.mean(quality_scores) if quality_scores else 0,
            'median': statistics.median(quality_scores) if quality_scores else 0,
            'distribution': {
                'excellent': sum(1 for score in quality_scores if score >= 0.8),
                'good': sum(1 for score in quality_scores if 0.6 <= score < 0.8),
                'fair': sum(1 for score in quality_scores if 0.4 <= score < 0.6),
                'poor': sum(1 for score in quality_scores if score < 0.4)
            }
        }
        
        # Topic analysis
        topic_analysis = await self._analyze_topics(content_list)
        
        # Entity analysis
        entity_analysis = await self._analyze_entities(content_list)
        
        # Temporal analysis
        temporal_analysis = await self._analyze_temporal_patterns(content_list)
        
        # Gap analysis
        gap_analysis = await self._identify_knowledge_gaps(content_list)
        
        return {
            'overview': {
                'total_items': total_items,
                'total_words': total_words,
                'average_words_per_item': total_words / max(total_items, 1),
                'content_types': dict(content_types.most_common())
            },
            'quality_analysis': quality_stats,
            'topic_analysis': topic_analysis,
            'entity_analysis': entity_analysis,
            'temporal_analysis': temporal_analysis,
            'gap_analysis': gap_analysis,
            'recommendations': await self._generate_improvement_recommendations(
                quality_stats, topic_analysis, gap_analysis
            )
        }
    
    async def _analyze_topics(self, content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze topic distribution"""
        topics = Counter()
        keywords = Counter()
        
        for item in content_list:
            # Extract keywords from text
            text = item.get('text', '')
            if text:
                doc = self.nlp(text)
                item_keywords = [token.lemma_.lower() for token in doc 
                               if not token.is_stop and not token.is_punct and token.is_alpha and len(token.text) > 2]
                keywords.update(item_keywords)
            
            # Get declared topics
            metadata = item.get('metadata', {})
            topic = metadata.get('topic') or metadata.get('category')
            if topic:
                topics[topic] += 1
        
        return {
            'declared_topics': dict(topics.most_common(10)),
            'top_keywords': dict(keywords.most_common(20)),
            'topic_coverage': len(topics),
            'keyword_diversity': len(keywords)
        }
    
    async def _analyze_entities(self, content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze entity distribution"""
        entity_types = Counter()
        entity_names = Counter()
        
        for item in content_list:
            text = item.get('text', '')
            if text:
                doc = self.nlp(text)
                for ent in doc.ents:
                    entity_types[ent.label_] += 1
                    entity_names[ent.text] += 1
        
        return {
            'entity_types': dict(entity_types.most_common()),
            'top_entities': dict(entity_names.most_common(20)),
            'total_unique_entities': len(entity_names)
        }
    
    async def _analyze_temporal_patterns(self, content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in content"""
        dates = []
        
        for item in content_list:
            metadata = item.get('metadata', {})
            date_str = metadata.get('created_at') or metadata.get('date')
            
            if date_str:
                try:
                    if isinstance(date_str, str):
                        date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    else:
                        date = date_str
                    dates.append(date.replace(tzinfo=None))
                except Exception:
                    continue
        
        if not dates:
            return {'error': 'No valid dates found'}
        
        # Analyze date distribution
        date_range = max(dates) - min(dates)
        monthly_counts = Counter()
        yearly_counts = Counter()
        
        for date in dates:
            monthly_counts[f"{date.year}-{date.month:02d}"] += 1
            yearly_counts[date.year] += 1
        
        return {
            'date_range_days': date_range.days,
            'oldest_content': min(dates).isoformat(),
            'newest_content': max(dates).isoformat(),
            'monthly_distribution': dict(monthly_counts.most_common(12)),
            'yearly_distribution': dict(yearly_counts.most_common()),
            'content_age_stats': {
                'avg_age_days': statistics.mean([(datetime.utcnow() - date).days for date in dates]),
                'median_age_days': statistics.median([(datetime.utcnow() - date).days for date in dates])
            }
        }
    
    async def _identify_knowledge_gaps(self, content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify potential knowledge gaps"""
        # This is a simplified gap analysis
        # In a real system, this would be more sophisticated
        
        topics = Counter()
        entity_types = Counter()
        
        for item in content_list:
            # Count topics
            metadata = item.get('metadata', {})
            topic = metadata.get('topic') or metadata.get('category')
            if topic:
                topics[topic] += 1
            
            # Count entity types
            text = item.get('text', '')
            if text:
                doc = self.nlp(text)
                for ent in doc.ents:
                    entity_types[ent.label_] += 1
        
        # Identify underrepresented areas
        avg_topic_count = statistics.mean(topics.values()) if topics else 0
        underrepresented_topics = [topic for topic, count in topics.items() 
                                 if count < avg_topic_count * 0.5]
        
        return {
            'underrepresented_topics': underrepresented_topics,
            'missing_entity_types': [
                'MONEY', 'LAW', 'LANGUAGE'  # Common entity types that might be missing
            ] if not any(et in entity_types for et in ['MONEY', 'LAW', 'LANGUAGE']) else [],
            'suggestions': [
                f"Consider adding more content about: {topic}" for topic in underrepresented_topics[:5]
            ]
        }
    
    async def _generate_improvement_recommendations(self, quality_stats: Dict[str, Any],
                                                  topic_analysis: Dict[str, Any],
                                                  gap_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for knowledge base improvement"""
        recommendations = []
        
        # Quality-based recommendations
        if quality_stats['average'] < 0.6:
            recommendations.append("Overall content quality is below average - implement quality review process")
        
        poor_quality_ratio = quality_stats['distribution']['poor'] / sum(quality_stats['distribution'].values())
        if poor_quality_ratio > 0.2:
            recommendations.append("High proportion of poor quality content - prioritize content cleanup")
        
        # Topic diversity recommendations
        if topic_analysis['topic_coverage'] < 5:
            recommendations.append("Limited topic coverage - consider expanding content diversity")
        
        # Gap-based recommendations
        recommendations.extend(gap_analysis.get('suggestions', []))
        
        return recommendations[:10]  # Limit to top 10 recommendations


class KnowledgeCuratorAgent:
    """Main knowledge curator agent that manages quality and provides analytics"""
    
    def __init__(self, agent_id: str = "knowledge-curator-agent"):
        self.agent_id = agent_id
        self.client = AMPClient(agent_id)
        
        # Initialize components
        self.quality_analyzer = QualityAnalyzer()
        self.duplication_detector = DuplicationDetector()
        self.analytics = KnowledgeAnalytics()
        
        logger.info(f"Knowledge Curator Agent {agent_id} initialized")
    
    async def start(self, host: str = "localhost", port: int = 8000):
        """Start the agent and register capabilities"""
        await self.client.connect(f"ws://{host}:{port}/ws")
        
        # Register capabilities
        capabilities = [
            {
                "name": "quality-analysis",
                "description": "Analyze content quality and provide quality scores",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "object"},
                        "batch_content": {"type": "array"}
                    }
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "quality_analysis": {"type": "object"},
                        "recommendations": {"type": "array"}
                    }
                }
            },
            {
                "name": "duplicate-detection",
                "description": "Detect duplicate and near-duplicate content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content_list": {"type": "array"},
                        "similarity_threshold": {"type": "number", "default": 0.85}
                    },
                    "required": ["content_list"]
                }
            },
            {
                "name": "knowledge-analytics",
                "description": "Generate analytics and insights about knowledge base",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content_list": {"type": "array"},
                        "report_type": {"type": "string", "enum": ["overview", "detailed", "quality_focus"]}
                    },
                    "required": ["content_list"]
                }
            },
            {
                "name": "content-validation",
                "description": "Validate content against quality standards",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "object"},
                        "standards": {"type": "object"}
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "improvement-suggestions",
                "description": "Provide suggestions for knowledge base improvement",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_results": {"type": "object"}
                    }
                }
            }
        ]
        
        for capability in capabilities:
            await self.client.register_capability(capability)
        
        # Start message handling
        await self.client.start_message_handler(self._handle_message)
        logger.info(f"Knowledge Curator Agent started on {host}:{port}")
    
    async def _handle_message(self, message: AMPMessage):
        """Handle incoming AMP messages"""
        try:
            capability = message.message.destination.capability
            payload = message.message.payload
            
            if capability == "quality-analysis":
                result = await self._handle_quality_analysis(payload)
            elif capability == "duplicate-detection":
                result = await self._handle_duplicate_detection(payload)
            elif capability == "knowledge-analytics":
                result = await self._handle_knowledge_analytics(payload)
            elif capability == "content-validation":
                result = await self._handle_content_validation(payload)
            elif capability == "improvement-suggestions":
                result = await self._handle_improvement_suggestions(payload)
            else:
                raise ValueError(f"Unknown capability: {capability}")
            
            # Send response
            await self.client.send_response(
                message.message.id,
                message.message.source.agent_id,
                result
            )
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self.client.send_error(
                message.message.id,
                message.message.source.agent_id,
                str(e),
                "CURATION_ERROR"
            )
    
    async def _handle_quality_analysis(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quality analysis request"""
        content = payload.get('content')
        batch_content = payload.get('batch_content')
        
        if batch_content:
            # Batch analysis
            results = []
            for item in batch_content:
                analysis = await self.quality_analyzer.analyze_content_quality(item)
                results.append({
                    'content_id': item.get('id'),
                    'quality_analysis': analysis
                })
            
            # Calculate batch statistics
            quality_scores = [r['quality_analysis']['quality_score'] for r in results]
            batch_stats = {
                'average_quality': statistics.mean(quality_scores),
                'median_quality': statistics.median(quality_scores),
                'quality_distribution': {
                    'excellent': sum(1 for score in quality_scores if score >= 0.8),
                    'good': sum(1 for score in quality_scores if 0.6 <= score < 0.8),
                    'fair': sum(1 for score in quality_scores if 0.4 <= score < 0.6),
                    'poor': sum(1 for score in quality_scores if score < 0.4)
                }
            }
            
            return {
                'batch_analysis': results,
                'batch_statistics': batch_stats,
                'total_analyzed': len(results)
            }
        
        elif content:
            # Single content analysis
            analysis = await self.quality_analyzer.analyze_content_quality(content)
            return {
                'quality_analysis': analysis,
                'content_id': content.get('id')
            }
        
        else:
            raise ValueError("Either 'content' or 'batch_content' must be provided")
    
    async def _handle_duplicate_detection(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle duplicate detection request"""
        content_list = payload['content_list']
        similarity_threshold = payload.get('similarity_threshold', 0.85)
        
        # Update detector threshold
        self.duplication_detector.similarity_threshold = similarity_threshold
        
        # Find duplicates
        duplicates = await self.duplication_detector.find_duplicates(content_list)
        
        return {
            'duplicates_found': len(duplicates),
            'duplicate_groups': duplicates,
            'similarity_threshold': similarity_threshold,
            'analysis_summary': {
                'total_items_analyzed': len(content_list),
                'duplicate_items': sum(len(group['items']) for group in duplicates),
                'unique_items': len(content_list) - sum(len(group['items']) for group in duplicates)
            }
        }
    
    async def _handle_knowledge_analytics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle knowledge analytics request"""
        content_list = payload['content_list']
        report_type = payload.get('report_type', 'overview')
        
        # Generate comprehensive report
        full_report = await self.analytics.generate_knowledge_base_report(content_list)
        
        if report_type == 'overview':
            # Return overview sections
            return {
                'report_type': report_type,
                'overview': full_report.get('overview', {}),
                'quality_summary': {
                    'average_quality': full_report.get('quality_analysis', {}).get('average', 0),
                    'distribution': full_report.get('quality_analysis', {}).get('distribution', {})
                },
                'top_topics': list(full_report.get('topic_analysis', {}).get('declared_topics', {}).keys())[:5],
                'key_recommendations': full_report.get('recommendations', [])[:3]
            }
        
        elif report_type == 'quality_focus':
            # Return quality-focused analysis
            return {
                'report_type': report_type,
                'quality_analysis': full_report.get('quality_analysis', {}),
                'quality_recommendations': [rec for rec in full_report.get('recommendations', []) 
                                          if 'quality' in rec.lower()]
            }
        
        else:  # detailed
            return {
                'report_type': report_type,
                'full_report': full_report
            }
    
    async def _handle_content_validation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle content validation request"""
        content = payload['content']
        standards = payload.get('standards', {})
        
        # Perform quality analysis
        quality_analysis = await self.quality_analyzer.analyze_content_quality(content)
        
        # Check against standards
        validation_results = {
            'passes_validation': True,
            'violations': [],
            'warnings': []
        }
        
        # Apply standards
        min_quality = standards.get('min_quality_score', 0.5)
        if quality_analysis['quality_score'] < min_quality:
            validation_results['passes_validation'] = False
            validation_results['violations'].append(
                f"Quality score {quality_analysis['quality_score']:.2f} below minimum {min_quality}"
            )
        
        min_words = standards.get('min_word_count', 50)
        word_count = len(content.get('text', '').split())
        if word_count < min_words:
            validation_results['passes_validation'] = False
            validation_results['violations'].append(
                f"Word count {word_count} below minimum {min_words}"
            )
        
        # Add warnings for issues
        for issue in quality_analysis.get('issues', []):
            validation_results['warnings'].append(issue)
        
        return {
            'validation_results': validation_results,
            'quality_analysis': quality_analysis,
            'content_id': content.get('id')
        }
    
    async def _handle_improvement_suggestions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle improvement suggestions request"""
        analysis_results = payload.get('analysis_results', {})
        
        suggestions = []
        
        # Extract suggestions from analysis results
        if 'quality_analysis' in analysis_results:
            quality = analysis_results['quality_analysis']
            if quality.get('average', 0) < 0.6:
                suggestions.append({
                    'category': 'Quality Improvement',
                    'priority': 'High',
                    'suggestion': 'Implement content review workflow for low-quality items',
                    'expected_impact': 'Significant improvement in overall knowledge base quality'
                })
        
        if 'gap_analysis' in analysis_results:
            gaps = analysis_results['gap_analysis']
            for suggestion in gaps.get('suggestions', []):
                suggestions.append({
                    'category': 'Content Gaps',
                    'priority': 'Medium',
                    'suggestion': suggestion,
                    'expected_impact': 'Better topic coverage and completeness'
                })
        
        # Add general suggestions
        suggestions.extend([
            {
                'category': 'Process Improvement',
                'priority': 'Medium',
                'suggestion': 'Implement regular duplicate detection and cleanup',
                'expected_impact': 'Reduced redundancy and improved search relevance'
            },
            {
                'category': 'Quality Assurance',
                'priority': 'Medium',
                'suggestion': 'Establish content freshness review schedule',
                'expected_impact': 'Improved accuracy and relevance of information'
            }
        ])
        
        return {
            'improvement_suggestions': suggestions,
            'total_suggestions': len(suggestions),
            'high_priority_count': sum(1 for s in suggestions if s['priority'] == 'High')
        }
    
    async def stop(self):
        """Stop the agent"""
        await self.client.disconnect()
        logger.info("Knowledge Curator Agent stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Curator Agent")
    parser.add_argument("--host", default="localhost", help="Host to connect to")
    parser.add_argument("--port", type=int, default=8000, help="Port to connect to")
    parser.add_argument("--agent-id", default="knowledge-curator-agent", help="Agent ID")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    agent = KnowledgeCuratorAgent(args.agent_id)
    
    try:
        asyncio.run(agent.start(args.host, args.port))
    except KeyboardInterrupt:
        asyncio.run(agent.stop())