"""
Synthesis Agent for Research Assistant Network

Handles content synthesis, report generation, and citation management.
Integrates with CrewAI for coordinated research workflows.
"""

import asyncio
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json

# NLP and text processing
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer

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
class SourceInfo:
    """Information about a source used in synthesis."""
    url: str
    title: str
    author: Optional[str] = None
    publication_date: Optional[str] = None
    credibility_score: float = 0.0
    source_type: str = "unknown"  # academic, news, government, etc.
    key_content: str = ""
    citation_format: str = ""


@dataclass
class SynthesisSection:
    """A section of synthesized content."""
    title: str
    content: str
    sources: List[SourceInfo]
    confidence_score: float
    section_type: str = "general"  # introduction, methodology, findings, conclusion


@dataclass
class SynthesisRequest:
    """Request for content synthesis."""
    topic: str
    source_materials: List[Dict[str, Any]]
    report_format: str = "academic"  # academic, journalistic, executive_summary
    target_length: int = 1500
    include_citations: bool = True
    focus_areas: List[str] = field(default_factory=list)
    quality_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisResult:
    """Result of content synthesis."""
    title: str
    executive_summary: str
    sections: List[SynthesisSection]
    conclusions: List[str]
    recommendations: List[str]
    citations: List[SourceInfo]
    metadata: Dict[str, Any]
    word_count: int
    quality_score: float


class SynthesisAgent:
    """Synthesis agent that combines verified information into coherent reports."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.SynthesisAgent")
        
        # Initialize NLP models
        self._initialize_nlp_models()
        
        # Report templates
        self.report_templates = self._load_report_templates()
        
        # Citation styles
        self.citation_styles = self._initialize_citation_styles()
        
        # Configuration
        self.default_target_length = self.config.get("default_target_length", 1500)
        self.min_sources_per_section = self.config.get("min_sources_per_section", 2)
        self.max_sections = self.config.get("max_sections", 8)
        
        # AMP client
        self.amp_client: Optional[AMPClient] = None
        
        # CrewAI agent
        self.crew_agent = self._create_crew_agent()
    
    def _initialize_nlp_models(self):
        """Initialize NLP models for synthesis."""
        try:
            # Initialize text generation model
            self.text_generator = pipeline(
                "text-generation",
                model="gpt2-medium",
                device=0 if torch.cuda.is_available() else -1,
                pad_token_id=50256
            )
            self.logger.info("Loaded GPT-2 text generation model")
        except Exception as e:
            self.logger.warning(f"Failed to load text generation model: {e}")
            self.text_generator = None
        
        try:
            # Initialize sentence transformer for content organization
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Loaded SentenceTransformer model")
        except Exception as e:
            self.logger.warning(f"Failed to load SentenceTransformer: {e}")
            self.sentence_model = None
    
    def _load_report_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load report templates for different formats."""
        return {
            "academic": {
                "sections": ["introduction", "methodology", "findings", "discussion", "conclusion"],
                "citation_style": "apa",
                "tone": "formal",
                "structure": "structured"
            },
            "journalistic": {
                "sections": ["lead", "background", "key_findings", "implications", "expert_opinions"],
                "citation_style": "news",
                "tone": "accessible",
                "structure": "inverted_pyramid"
            },
            "executive_summary": {
                "sections": ["overview", "key_insights", "recommendations", "next_steps"],
                "citation_style": "minimal",
                "tone": "business",
                "structure": "bullet_points"
            },
            "technical_report": {
                "sections": ["abstract", "introduction", "technical_details", "analysis", "conclusions"],
                "citation_style": "ieee",
                "tone": "technical",
                "structure": "detailed"
            }
        }
    
    def _initialize_citation_styles(self) -> Dict[str, Dict[str, str]]:
        """Initialize citation style formats."""
        return {
            "apa": {
                "format": "{author} ({year}). {title}. {source}. Retrieved from {url}",
                "in_text": "({author}, {year})"
            },
            "mla": {
                "format": "{author}. \"{title}.\" {source}, {date}. Web. {access_date}.",
                "in_text": "({author})"
            },
            "ieee": {
                "format": "[{number}] {author}, \"{title},\" {source}, {date}. [Online]. Available: {url}",
                "in_text": "[{number}]"
            },
            "news": {
                "format": "{source} report on {topic}",
                "in_text": "according to {source}"
            },
            "minimal": {
                "format": "{source}",
                "in_text": "({source})"
            }
        }
    
    def _create_crew_agent(self) -> Agent:
        """Create the CrewAI agent for synthesis."""
        return Agent(
            role="Research Synthesis Specialist",
            goal="Create comprehensive, well-structured reports by synthesizing information from multiple sources",
            backstory="""You are an expert research synthesizer with extensive experience in 
            academic writing, content organization, and source integration. You excel at 
            identifying key themes, organizing complex information, and creating coherent 
            narratives that combine insights from multiple sources while maintaining 
            academic rigor and clarity.""",
            tools=[],  # We'll add custom tools
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.3)  # Slightly higher for creative synthesis
        )
    
    async def synthesize_content(self, request: SynthesisRequest) -> SynthesisResult:
        """Synthesize content from multiple sources into a coherent report."""
        self.logger.info(f"Starting synthesis for topic: {request.topic}")
        
        # Prepare source materials
        processed_sources = self._process_source_materials(request.source_materials)
        
        # Organize content by themes and sections
        organized_content = await self._organize_content_by_themes(
            processed_sources, request.topic, request.focus_areas
        )
        
        # Generate report structure
        report_structure = self._generate_report_structure(
            request.report_format, organized_content, request.target_length
        )
        
        # Synthesize each section
        sections = []
        for section_info in report_structure["sections"]:
            section = await self._synthesize_section(
                section_info, organized_content, request
            )
            sections.append(section)
        
        # Generate executive summary
        executive_summary = await self._generate_executive_summary(
            sections, request.topic, request.target_length
        )
        
        # Extract conclusions and recommendations
        conclusions = self._extract_conclusions(sections)
        recommendations = self._generate_recommendations(sections, request.topic)
        
        # Prepare citations
        all_citations = self._prepare_citations(
            processed_sources, request.report_format
        )
        
        # Calculate quality score
        quality_score = self._calculate_synthesis_quality(
            sections, processed_sources, request
        )
        
        # Prepare metadata
        metadata = {
            "synthesis_timestamp": datetime.now().isoformat(),
            "topic": request.topic,
            "report_format": request.report_format,
            "source_count": len(processed_sources),
            "focus_areas": request.focus_areas,
            "quality_requirements": request.quality_requirements
        }
        
        result = SynthesisResult(
            title=self._generate_title(request.topic, request.report_format),
            executive_summary=executive_summary,
            sections=sections,
            conclusions=conclusions,
            recommendations=recommendations,
            citations=all_citations,
            metadata=metadata,
            word_count=sum(len(section.content.split()) for section in sections),
            quality_score=quality_score
        )
        
        self.logger.info(f"Synthesis complete: {result.word_count} words, quality score: {quality_score:.2f}")
        return result
    
    def _process_source_materials(self, source_materials: List[Dict[str, Any]]) -> List[SourceInfo]:
        """Process and structure source materials."""
        processed_sources = []
        
        for material in source_materials:
            try:
                source = SourceInfo(
                    url=material.get("url", ""),
                    title=material.get("title", "Untitled"),
                    author=material.get("author"),
                    publication_date=material.get("publication_date"),
                    credibility_score=material.get("credibility_score", 0.5),
                    source_type=material.get("source_type", "unknown"),
                    key_content=material.get("content", material.get("summary", "")),
                    citation_format=""
                )
                
                if source.key_content:  # Only include sources with content
                    processed_sources.append(source)
                    
            except Exception as e:
                self.logger.warning(f"Failed to process source material: {e}")
        
        # Sort by credibility score
        processed_sources.sort(key=lambda x: x.credibility_score, reverse=True)
        
        return processed_sources
    
    async def _organize_content_by_themes(self, sources: List[SourceInfo], 
                                        topic: str, focus_areas: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Organize content into thematic groups."""
        if not self.sentence_model:
            return await self._organize_content_by_keywords(sources, topic, focus_areas)
        
        # Extract key sentences from each source
        all_sentences = []
        sentence_to_source = {}
        
        for source in sources:
            sentences = self._extract_key_sentences(source.key_content)
            for sentence in sentences:
                sentence_info = {
                    "text": sentence,
                    "source": source,
                    "embedding": None
                }
                all_sentences.append(sentence_info)
                sentence_to_source[sentence] = source
        
        if not all_sentences:
            return {}
        
        # Generate embeddings for clustering
        sentence_texts = [s["text"] for s in all_sentences]
        embeddings = self.sentence_model.encode(sentence_texts)
        
        # Simple clustering based on similarity
        themes = defaultdict(list)
        processed = set()
        
        for i, sentence_info in enumerate(all_sentences):
            if i in processed:
                continue
            
            # Start a new theme
            theme_sentences = [sentence_info]
            processed.add(i)
            
            # Find similar sentences
            for j, other_sentence in enumerate(all_sentences):
                if j <= i or j in processed:
                    continue
                
                similarity = self._calculate_cosine_similarity(embeddings[i], embeddings[j])
                if similarity > 0.6:  # Similarity threshold
                    theme_sentences.append(other_sentence)
                    processed.add(j)
            
            # Create theme name based on most common keywords
            theme_name = self._generate_theme_name(theme_sentences, topic)
            themes[theme_name] = theme_sentences
        
        return dict(themes)
    
    async def _organize_content_by_keywords(self, sources: List[SourceInfo], 
                                          topic: str, focus_areas: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Fallback content organization using keyword matching."""
        themes = defaultdict(list)
        
        # Define theme keywords
        theme_keywords = {
            "background": ["background", "history", "context", "introduction", "overview"],
            "methodology": ["method", "approach", "technique", "procedure", "process"],
            "findings": ["result", "finding", "discovery", "outcome", "data"],
            "analysis": ["analysis", "interpretation", "discussion", "implication"],
            "challenges": ["challenge", "problem", "issue", "difficulty", "limitation"],
            "benefits": ["benefit", "advantage", "positive", "improvement", "success"],
            "future": ["future", "trend", "prediction", "outlook", "prospect"]
        }
        
        # Add focus areas as themes
        for area in focus_areas:
            theme_keywords[area.lower()] = [area.lower()]
        
        for source in sources:
            content_lower = source.key_content.lower()
            sentences = self._extract_key_sentences(source.key_content)
            
            for theme, keywords in theme_keywords.items():
                matching_sentences = []
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in keywords):
                        matching_sentences.append({
                            "text": sentence,
                            "source": source
                        })
                
                if matching_sentences:
                    themes[theme].extend(matching_sentences)
        
        return dict(themes)
    
    def _extract_key_sentences(self, content: str, max_sentences: int = 10) -> List[str]:
        """Extract key sentences from content."""
        if not content:
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Score sentences based on various factors
        scored_sentences = []
        for sentence in sentences:
            score = self._score_sentence_importance(sentence, content)
            scored_sentences.append((sentence, score))
        
        # Sort by score and return top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sentence for sentence, score in scored_sentences[:max_sentences]]
    
    def _score_sentence_importance(self, sentence: str, full_content: str) -> float:
        """Score a sentence's importance within the content."""
        score = 0.0
        
        # Length factor (moderate length preferred)
        words = sentence.split()
        if 10 <= len(words) <= 30:
            score += 1.0
        elif 5 <= len(words) <= 40:
            score += 0.5
        
        # Important keywords
        important_keywords = [
            'study', 'research', 'found', 'shows', 'indicates', 'reveals',
            'important', 'significant', 'key', 'main', 'crucial', 'essential',
            'result', 'conclusion', 'finding', 'discovery'
        ]
        
        sentence_lower = sentence.lower()
        for keyword in important_keywords:
            if keyword in sentence_lower:
                score += 0.5
        
        # Numerical data bonus
        if re.search(r'\b\d+(?:\.\d+)?%?\b', sentence):
            score += 0.3
        
        # Position bonus (earlier sentences often more important)
        position_in_content = full_content.find(sentence)
        if position_in_content != -1:
            relative_position = position_in_content / len(full_content)
            if relative_position < 0.3:  # First 30% of content
                score += 0.2
        
        return score
    
    def _generate_report_structure(self, report_format: str, 
                                 organized_content: Dict[str, List], 
                                 target_length: int) -> Dict[str, Any]:
        """Generate the structure for the report."""
        template = self.report_templates.get(report_format, self.report_templates["academic"])
        
        # Calculate target length per section
        base_sections = template["sections"]
        target_per_section = target_length // len(base_sections)
        
        sections = []
        available_themes = list(organized_content.keys())
        
        for section_name in base_sections:
            # Map section to available themes
            relevant_themes = self._map_section_to_themes(section_name, available_themes)
            
            section_info = {
                "name": section_name,
                "target_length": target_per_section,
                "themes": relevant_themes,
                "priority": self._get_section_priority(section_name)
            }
            sections.append(section_info)
        
        return {
            "sections": sections,
            "template": template,
            "total_target_length": target_length
        }
    
    def _map_section_to_themes(self, section_name: str, available_themes: List[str]) -> List[str]:
        """Map report sections to available content themes."""
        section_theme_mapping = {
            "introduction": ["background", "overview", "context"],
            "methodology": ["methodology", "approach", "method"],
            "findings": ["findings", "results", "data", "discovery"],
            "discussion": ["analysis", "implications", "interpretation"],
            "conclusion": ["conclusions", "summary", "outcomes"],
            "background": ["background", "history", "context"],
            "key_findings": ["findings", "results", "discovery"],
            "implications": ["implications", "analysis", "future"],
            "recommendations": ["recommendations", "suggestions", "next_steps"],
            "overview": ["background", "summary", "introduction"],
            "key_insights": ["insights", "findings", "analysis"],
            "next_steps": ["future", "recommendations", "actions"]
        }
        
        preferred_themes = section_theme_mapping.get(section_name.lower(), [section_name.lower()])
        
        # Find matching themes
        relevant_themes = []
        for theme in available_themes:
            theme_lower = theme.lower()
            if any(preferred in theme_lower or theme_lower in preferred 
                  for preferred in preferred_themes):
                relevant_themes.append(theme)
        
        # If no specific matches, include all themes for comprehensive coverage
        if not relevant_themes and available_themes:
            relevant_themes = available_themes[:3]  # Limit to top 3 themes
        
        return relevant_themes
    
    def _get_section_priority(self, section_name: str) -> int:
        """Get priority score for section (higher = more important)."""
        priority_mapping = {
            "introduction": 9,
            "overview": 9,
            "key_findings": 10,
            "findings": 10,
            "results": 10,
            "analysis": 8,
            "discussion": 8,
            "conclusion": 9,
            "recommendations": 7,
            "methodology": 6,
            "background": 5,
            "implications": 7,
            "next_steps": 6
        }
        
        return priority_mapping.get(section_name.lower(), 5)
    
    async def _synthesize_section(self, section_info: Dict[str, Any], 
                                organized_content: Dict[str, List], 
                                request: SynthesisRequest) -> SynthesisSection:
        """Synthesize content for a specific section."""
        section_name = section_info["name"]
        target_length = section_info["target_length"]
        relevant_themes = section_info["themes"]
        
        # Collect relevant content
        section_content = []
        section_sources = set()
        
        for theme in relevant_themes:
            if theme in organized_content:
                theme_content = organized_content[theme]
                section_content.extend(theme_content)
                for item in theme_content:
                    section_sources.add(item["source"])
        
        if not section_content:
            # Create minimal section if no content available
            return SynthesisSection(
                title=section_name.title(),
                content=f"Further research is needed in this area of {request.topic}.",
                sources=list(section_sources),
                confidence_score=0.1,
                section_type=section_name.lower()
            )
        
        # Generate section content
        synthesized_text = await self._generate_section_text(
            section_name, section_content, target_length, request
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_section_confidence(
            section_content, list(section_sources)
        )
        
        return SynthesisSection(
            title=self._format_section_title(section_name),
            content=synthesized_text,
            sources=list(section_sources),
            confidence_score=confidence_score,
            section_type=section_name.lower()
        )
    
    async def _generate_section_text(self, section_name: str, 
                                   content_items: List[Dict[str, Any]], 
                                   target_length: int,
                                   request: SynthesisRequest) -> str:
        """Generate synthesized text for a section."""
        
        # Collect and organize content
        key_points = []
        for item in content_items:
            key_points.append(item["text"])
        
        if not key_points:
            return f"No specific information is available for {section_name.lower()} regarding {request.topic}."
        
        # Sort key points by source credibility and relevance
        scored_points = []
        for item in content_items:
            score = item["source"].credibility_score
            scored_points.append((item["text"], score))
        
        scored_points.sort(key=lambda x: x[1], reverse=True)
        top_points = [point for point, score in scored_points[:8]]  # Top 8 points
        
        # Generate synthesized text
        if self.text_generator and len(" ".join(top_points)) < 800:
            try:
                synthesized = await self._generate_with_ai(
                    section_name, top_points, target_length, request.topic
                )
                if synthesized and len(synthesized) > 100:
                    return synthesized
            except Exception as e:
                self.logger.warning(f"AI text generation failed: {e}")
        
        # Fallback: manual synthesis
        return self._manual_synthesis(section_name, top_points, target_length, request.topic)
    
    async def _generate_with_ai(self, section_name: str, key_points: List[str], 
                              target_length: int, topic: str) -> str:
        """Generate text using AI model."""
        # Prepare prompt
        points_text = ". ".join(key_points[:5])  # Use top 5 points
        prompt = f"""Write a {section_name} section about {topic}. 
        
        Key information to include:
        {points_text}
        
        Write approximately {target_length//6} words in a clear, academic style:"""
        
        # Generate text
        try:
            generated = self.text_generator(
                prompt,
                max_length=len(prompt.split()) + target_length//4,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
            
            generated_text = generated[0]['generated_text']
            
            # Extract only the generated part (after the prompt)
            generated_part = generated_text[len(prompt):].strip()
            
            # Clean up the generated text
            sentences = generated_part.split('.')
            clean_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10 and not sentence.startswith('The following'):
                    clean_sentences.append(sentence)
                
                # Stop if we have enough content
                if len('. '.join(clean_sentences)) >= target_length * 0.8:
                    break
            
            return '. '.join(clean_sentences) + '.'
            
        except Exception as e:
            self.logger.warning(f"AI generation failed: {e}")
            return ""
    
    def _manual_synthesis(self, section_name: str, key_points: List[str], 
                         target_length: int, topic: str) -> str:
        """Manually synthesize content from key points."""
        if not key_points:
            return f"Further research is needed regarding {section_name.lower()} of {topic}."
        
        # Group similar points
        grouped_points = self._group_similar_points(key_points)
        
        # Generate paragraphs
        paragraphs = []
        
        # Introduction sentence
        intro_templates = {
            "introduction": f"Research on {topic} reveals several important aspects.",
            "findings": f"Key findings regarding {topic} include the following insights.",
            "analysis": f"Analysis of available data on {topic} demonstrates several patterns.",
            "conclusion": f"Based on the available evidence, {topic} presents several conclusions.",
            "methodology": f"The research on {topic} employed various methodological approaches.",
            "background": f"The background of {topic} encompasses several key developments."
        }
        
        intro = intro_templates.get(section_name.lower(), 
                                   f"Examination of {topic} reveals important information.")
        paragraphs.append(intro)
        
        # Process grouped points
        for group in grouped_points[:4]:  # Max 4 groups to control length
            if len(group) == 1:
                paragraphs.append(group[0])
            else:
                # Combine similar points
                combined = self._combine_similar_points(group)
                paragraphs.append(combined)
        
        # Combine paragraphs
        full_text = " ".join(paragraphs)
        
        # Adjust length if needed
        if len(full_text.split()) > target_length * 1.2:
            # Trim to target length
            words = full_text.split()
            trimmed_words = words[:target_length]
            full_text = " ".join(trimmed_words)
            
            # Ensure it ends with a complete sentence
            if not full_text.endswith('.'):
                last_period = full_text.rfind('.')
                if last_period > len(full_text) * 0.8:  # If period is reasonably close to end
                    full_text = full_text[:last_period + 1]
        
        return full_text
    
    def _group_similar_points(self, points: List[str]) -> List[List[str]]:
        """Group similar points together."""
        if not self.sentence_model:
            # Simple keyword-based grouping
            return [[point] for point in points[:5]]  # Just take top 5 as individual groups
        
        try:
            # Use semantic similarity for grouping
            embeddings = self.sentence_model.encode(points)
            
            groups = []
            processed = set()
            
            for i, point in enumerate(points):
                if i in processed:
                    continue
                
                group = [point]
                processed.add(i)
                
                # Find similar points
                for j, other_point in enumerate(points):
                    if j <= i or j in processed:
                        continue
                    
                    similarity = self._calculate_cosine_similarity(embeddings[i], embeddings[j])
                    if similarity > 0.7:  # High similarity threshold
                        group.append(other_point)
                        processed.add(j)
                
                groups.append(group)
            
            return groups[:4]  # Limit to 4 groups
            
        except Exception as e:
            self.logger.warning(f"Semantic grouping failed: {e}")
            return [[point] for point in points[:5]]
    
    def _combine_similar_points(self, points: List[str]) -> str:
        """Combine similar points into a coherent paragraph."""
        if len(points) == 1:
            return points[0]
        
        # Extract common themes and unique details
        all_words = []
        for point in points:
            all_words.extend(point.lower().split())
        
        # Find common words (excluding stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        word_freq = {}
        for word in all_words:
            if word not in stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Create combined text
        if len(points) <= 3:
            # Simple concatenation with transitions
            transitions = ["Additionally,", "Furthermore,", "Moreover,"]
            combined_parts = [points[0]]
            
            for i, point in enumerate(points[1:], 1):
                if i <= len(transitions):
                    combined_parts.append(f"{transitions[i-1]} {point.lower()}")
                else:
                    combined_parts.append(point)
            
            return " ".join(combined_parts)
        else:
            # Summarize multiple points
            key_info = []
            for point in points[:3]:  # Take top 3 points
                # Extract key phrase from each point
                sentences = point.split('.')
                if sentences:
                    key_info.append(sentences[0].strip())
            
            return f"Research indicates that {', '.join(key_info[:2])}, and {key_info[2] if len(key_info) > 2 else 'additional factors are relevant'}."
    
    async def _generate_executive_summary(self, sections: List[SynthesisSection], 
                                        topic: str, target_length: int) -> str:
        """Generate an executive summary from all sections."""
        # Extract key points from each section
        key_insights = []
        
        for section in sections:
            if section.confidence_score > 0.3:  # Only include confident sections
                # Extract first sentence or two from each section
                sentences = section.content.split('.')
                if sentences:
                    key_insights.append(sentences[0].strip())
        
        if not key_insights:
            return f"This report examines various aspects of {topic} based on available research and evidence."
        
        # Limit summary length
        summary_target = min(300, target_length // 5)
        
        # Create executive summary
        summary_parts = [
            f"This report provides a comprehensive analysis of {topic}.",
        ]
        
        # Add top insights
        for insight in key_insights[:4]:  # Top 4 insights
            if len(' '.join(summary_parts)) + len(insight) < summary_target:
                summary_parts.append(insight + '.')
        
        return ' '.join(summary_parts)
    
    def _extract_conclusions(self, sections: List[SynthesisSection]) -> List[str]:
        """Extract conclusions from synthesized sections."""
        conclusions = []
        
        # Look for conclusion-type sections
        conclusion_sections = [s for s in sections if 'conclusion' in s.section_type.lower()]
        
        for section in conclusion_sections:
            sentences = section.content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if (len(sentence) > 20 and 
                    any(indicator in sentence.lower() for indicator in 
                        ['therefore', 'thus', 'in conclusion', 'shows that', 'indicates that'])):
                    conclusions.append(sentence)
        
        # If no specific conclusions found, extract from high-confidence sections
        if not conclusions:
            for section in sections:
                if section.confidence_score > 0.7:
                    sentences = section.content.split('.')
                    if sentences:
                        conclusions.append(sentences[-1].strip())  # Last sentence often conclusive
        
        return conclusions[:5]  # Limit to 5 conclusions
    
    def _generate_recommendations(self, sections: List[SynthesisSection], topic: str) -> List[str]:
        """Generate recommendations based on synthesized content."""
        recommendations = []
        
        # Look for explicit recommendations in content
        for section in sections:
            content_lower = section.content.lower()
            if any(indicator in content_lower for indicator in 
                  ['recommend', 'suggest', 'should', 'could', 'would benefit']):
                sentences = section.content.split('.')
                for sentence in sentences:
                    if any(indicator in sentence.lower() for indicator in 
                          ['recommend', 'suggest', 'should']):
                        recommendations.append(sentence.strip())
        
        # Generate generic recommendations if none found
        if not recommendations:
            recommendations = [
                f"Further research on {topic} is recommended to expand current understanding.",
                f"Continued monitoring of developments in {topic} would be beneficial.",
                f"Stakeholders should consider the implications of {topic} for their operations."
            ]
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _prepare_citations(self, sources: List[SourceInfo], report_format: str) -> List[SourceInfo]:
        """Prepare properly formatted citations."""
        citation_style = self.report_templates.get(report_format, {}).get("citation_style", "apa")
        style_template = self.citation_styles.get(citation_style, self.citation_styles["apa"])
        
        for source in sources:
            try:
                # Extract information for citation
                author = source.author or "Unknown Author"
                year = self._extract_year(source.publication_date) or "n.d."
                title = source.title or "Untitled"
                domain = source.url.split('/')[2] if '/' in source.url else source.url
                
                # Format citation
                citation = style_template["format"].format(
                    author=author,
                    year=year,
                    title=title,
                    source=domain,
                    url=source.url,
                    number=len(sources)  # For numbered styles
                )
                
                source.citation_format = citation
                
            except Exception as e:
                self.logger.warning(f"Failed to format citation for {source.url}: {e}")
                source.citation_format = f"{source.title}. Retrieved from {source.url}"
        
        return sources
    
    def _extract_year(self, date_string: Optional[str]) -> Optional[str]:
        """Extract year from date string."""
        if not date_string:
            return None
        
        # Look for 4-digit year
        year_match = re.search(r'\b(19|20)\d{2}\b', date_string)
        return year_match.group() if year_match else None
    
    def _calculate_synthesis_quality(self, sections: List[SynthesisSection], 
                                   sources: List[SourceInfo], 
                                   request: SynthesisRequest) -> float:
        """Calculate overall quality score for the synthesis."""
        if not sections:
            return 0.0
        
        # Component scores
        scores = []
        
        # Source quality (30%)
        if sources:
            avg_source_credibility = sum(s.credibility_score for s in sources) / len(sources)
            scores.append(("source_quality", avg_source_credibility, 0.3))
        
        # Section confidence (25%)
        avg_section_confidence = sum(s.confidence_score for s in sections) / len(sections)
        scores.append(("section_confidence", avg_section_confidence, 0.25))
        
        # Content coverage (20%)
        total_words = sum(len(s.content.split()) for s in sections)
        length_score = min(1.0, total_words / request.target_length)
        scores.append(("content_coverage", length_score, 0.2))
        
        # Source diversity (15%)
        unique_domains = len(set(s.url.split('/')[2] if '/' in s.url else s.url for s in sources))
        diversity_score = min(1.0, unique_domains / max(1, len(sources) * 0.7))
        scores.append(("source_diversity", diversity_score, 0.15))
        
        # Structure completeness (10%)
        expected_sections = len(self.report_templates.get(request.report_format, {}).get("sections", []))
        structure_score = min(1.0, len(sections) / max(1, expected_sections))
        scores.append(("structure_completeness", structure_score, 0.1))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in scores)
        
        return min(1.0, total_score)
    
    def _generate_title(self, topic: str, report_format: str) -> str:
        """Generate an appropriate title for the report."""
        format_styles = {
            "academic": f"A Comprehensive Analysis of {topic}",
            "journalistic": f"{topic}: Key Findings and Implications",
            "executive_summary": f"Executive Summary: {topic}",
            "technical_report": f"Technical Report on {topic}"
        }
        
        return format_styles.get(report_format, f"Report on {topic}")
    
    def _format_section_title(self, section_name: str) -> str:
        """Format section title appropriately."""
        return ' '.join(word.capitalize() for word in section_name.split('_'))
    
    def _calculate_cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import numpy as np
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_section_confidence(self, content_items: List[Dict[str, Any]], 
                                    sources: List[SourceInfo]) -> float:
        """Calculate confidence score for a section."""
        if not content_items or not sources:
            return 0.0
        
        # Average source credibility
        avg_credibility = sum(s.credibility_score for s in sources) / len(sources)
        
        # Content quantity factor
        content_factor = min(1.0, len(content_items) / 3.0)  # Ideal: 3+ content items
        
        # Source diversity factor
        unique_sources = len(set(s.url for s in sources))
        diversity_factor = min(1.0, unique_sources / max(1, len(sources) * 0.8))
        
        # Combined confidence
        confidence = (avg_credibility * 0.6 + content_factor * 0.25 + diversity_factor * 0.15)
        
        return min(1.0, confidence)
    
    # AMP capability handlers
    async def handle_content_synthesis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle content synthesis capability requests."""
        topic = parameters.get("topic", "")
        source_materials = parameters.get("source_materials", [])
        report_format = parameters.get("report_format", "academic")
        target_length = parameters.get("target_length", self.default_target_length)
        include_citations = parameters.get("include_citations", True)
        focus_areas = parameters.get("focus_areas", [])
        
        if not topic or not source_materials:
            return {"error": "Topic and source materials are required", "result": None}
        
        request = SynthesisRequest(
            topic=topic,
            source_materials=source_materials,
            report_format=report_format,
            target_length=target_length,
            include_citations=include_citations,
            focus_areas=focus_areas
        )
        
        result = await self.synthesize_content(request)
        
        return {
            "result": {
                "title": result.title,
                "executive_summary": result.executive_summary,
                "sections": [
                    {
                        "title": section.title,
                        "content": section.content,
                        "section_type": section.section_type,
                        "confidence_score": section.confidence_score,
                        "source_count": len(section.sources)
                    }
                    for section in result.sections
                ],
                "conclusions": result.conclusions,
                "recommendations": result.recommendations,
                "citations": [
                    {
                        "title": citation.title,
                        "url": citation.url,
                        "author": citation.author,
                        "publication_date": citation.publication_date,
                        "credibility_score": citation.credibility_score,
                        "citation_format": citation.citation_format
                    }
                    for citation in result.citations
                ] if include_citations else [],
                "metadata": result.metadata,
                "word_count": result.word_count,
                "quality_score": result.quality_score
            },
            "success": True
        }
    
    async def handle_report_formatting(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle report formatting capability requests."""
        content = parameters.get("content", "")
        format_type = parameters.get("format", "academic")
        include_citations = parameters.get("include_citations", True)
        
        if not content:
            return {"error": "Content is required", "formatted_report": None}
        
        # This is a simplified formatting handler
        # In practice, you'd implement more sophisticated formatting
        
        template = self.report_templates.get(format_type, self.report_templates["academic"])
        
        formatted_content = {
            "format_type": format_type,
            "template_applied": template,
            "formatted_content": content,  # Would apply actual formatting here
            "citation_style": template.get("citation_style", "apa"),
            "formatting_timestamp": datetime.now().isoformat()
        }
        
        return {
            "formatted_report": formatted_content,
            "success": True
        }
    
    async def start_amp_agent(self, agent_id: str = "synthesis-agent",
                             endpoint: str = "http://localhost:8000") -> AMPClient:
        """Start the AMP agent."""
        
        self.amp_client = await (
            AMPBuilder(agent_id, "Synthesis Agent")
            .with_framework("crewai")
            .with_transport(TransportType.HTTP, endpoint)
            .add_capability(
                "content-synthesis",
                self.handle_content_synthesis,
                "Synthesize information from multiple sources into coherent reports",
                "synthesis",
                input_schema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "source_materials": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string"},
                                    "title": {"type": "string"},
                                    "content": {"type": "string"},
                                    "credibility_score": {"type": "number"},
                                    "source_type": {"type": "string"}
                                }
                            }
                        },
                        "report_format": {
                            "type": "string",
                            "enum": ["academic", "journalistic", "executive_summary", "technical_report"],
                            "default": "academic"
                        },
                        "target_length": {"type": "integer", "default": 1500},
                        "include_citations": {"type": "boolean", "default": True},
                        "focus_areas": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["topic", "source_materials"]
                },
                constraints=CapabilityConstraints(response_time_ms=45000)
            )
            .add_capability(
                "report-formatting",
                self.handle_report_formatting,
                "Format content according to specific report styles and citation standards",
                "formatting",
                input_schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "format": {
                            "type": "string",
                            "enum": ["academic", "journalistic", "executive_summary", "technical_report"],
                            "default": "academic"
                        },
                        "include_citations": {"type": "boolean", "default": True}
                    },
                    "required": ["content"]
                }
            )
            .build()
        )
        
        return self.amp_client


async def main():
    """Main function for testing the synthesis agent."""
    logging.basicConfig(level=logging.INFO)
    
    # Create synthesis agent
    synthesizer = SynthesisAgent()
    
    # Start AMP agent
    client = await synthesizer.start_amp_agent()
    
    try:
        print("Synthesis Agent started. Testing synthesis functionality...")
        
        # Test synthesis
        test_sources = [
            {
                "url": "https://example.com/study1",
                "title": "AI Research Breakthrough",
                "content": "Recent studies show that artificial intelligence systems have achieved significant improvements in natural language processing. The research demonstrates a 40% increase in accuracy compared to previous models.",
                "credibility_score": 0.85,
                "source_type": "academic"
            },
            {
                "url": "https://example.com/article2",
                "title": "Machine Learning Applications",
                "content": "Machine learning applications are expanding across various industries. Healthcare, finance, and transportation sectors are seeing major benefits from AI implementation.",
                "credibility_score": 0.75,
                "source_type": "news"
            },
            {
                "url": "https://example.com/report3",
                "title": "Future of AI",
                "content": "Experts predict that AI will continue to evolve rapidly. Future developments may include improved reasoning capabilities and better human-AI collaboration systems.",
                "credibility_score": 0.80,
                "source_type": "report"
            }
        ]
        
        request = SynthesisRequest(
            topic="Artificial Intelligence Developments",
            source_materials=test_sources,
            report_format="academic",
            target_length=800,
            focus_areas=["technology", "applications", "future"]
        )
        
        result = await synthesizer.synthesize_content(request)
        
        print(f"\nSynthesis Results:")
        print("=" * 60)
        print(f"Title: {result.title}")
        print(f"Word Count: {result.word_count}")
        print(f"Quality Score: {result.quality_score:.2f}")
        print(f"Number of Sections: {len(result.sections)}")
        
        print(f"\nExecutive Summary:")
        print(result.executive_summary)
        
        print(f"\nSections:")
        for section in result.sections:
            print(f"\n{section.title} (Confidence: {section.confidence_score:.2f})")
            print("-" * 40)
            print(section.content[:200] + "..." if len(section.content) > 200 else section.content)
        
        if result.conclusions:
            print(f"\nConclusions:")
            for conclusion in result.conclusions:
                print(f"- {conclusion}")
        
        if result.recommendations:
            print(f"\nRecommendations:")
            for recommendation in result.recommendations:
                print(f"- {recommendation}")
        
        print("\nSynthesis Agent is running. Press Ctrl+C to stop.")
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())