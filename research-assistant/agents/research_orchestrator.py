"""
Research Orchestrator for Research Assistant Network

Coordinates CrewAI crews and manages research workflows.
Integrates with AMP protocol for distributed agent coordination.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from langchain.llms import OpenAI

# Local agent imports
from web_search_agent import WebSearchAgent, SearchQuery
from content_analyzer import ContentAnalyzer, AnalysisRequest
from fact_checker import FactChecker, Claim
from synthesis_agent import SynthesisAgent, SynthesisRequest

# AMP imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))
from amp_client import AMPClient
from amp_types import Capability, CapabilityConstraints, TransportType
from amp_utils import AMPBuilder


@dataclass
class ResearchQuery:
    """Represents a research query with parameters."""
    query: str
    depth: str = "standard"  # basic, standard, comprehensive
    max_sources: int = 10
    focus_areas: List[str] = field(default_factory=list)
    report_format: str = "academic"
    target_length: int = 1500
    include_fact_checking: bool = True
    quality_threshold: float = 0.6


@dataclass
class ResearchProgress:
    """Tracks progress of research workflow."""
    query: ResearchQuery
    status: str = "started"  # started, searching, analyzing, fact_checking, synthesizing, completed, failed
    current_step: str = ""
    progress_percentage: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class ResearchResult:
    """Final result of research workflow."""
    query: ResearchQuery
    title: str
    executive_summary: str
    sections: List[Dict[str, Any]]
    conclusions: List[str]
    recommendations: List[str]
    sources: List[Dict[str, Any]]
    fact_check_results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    quality_score: float
    word_count: int


class ResearchOrchestrator:
    """Orchestrates research workflows using CrewAI crews and AMP agents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.ResearchOrchestrator")
        
        # Initialize agents
        self.web_search_agent = WebSearchAgent(config.get("web_search", {}))
        self.content_analyzer = ContentAnalyzer(config.get("content_analysis", {}))
        self.fact_checker = FactChecker(config.get("fact_checking", {}))
        self.synthesis_agent = SynthesisAgent(config.get("synthesis", {}))
        
        # Progress tracking
        self.active_research: Dict[str, ResearchProgress] = {}
        
        # CrewAI crews
        self.crews = {}
        self._initialize_crews()
        
        # Configuration
        self.max_concurrent_research = self.config.get("max_concurrent_research", 3)
        self.default_timeout = self.config.get("default_timeout", 1800)  # 30 minutes
        
        # AMP client
        self.amp_client: Optional[AMPClient] = None
    
    def _initialize_crews(self):
        """Initialize CrewAI crews for different research workflows."""
        
        # Main research crew
        self.crews["research"] = self._create_research_crew()
        
        # Fact-checking focused crew
        self.crews["fact_check"] = self._create_fact_check_crew()
        
        # Quick research crew for simple queries
        self.crews["quick_research"] = self._create_quick_research_crew()
    
    def _create_research_crew(self) -> Crew:
        """Create the main comprehensive research crew."""
        
        # Define CrewAI agents (wrappers around our AMP agents)
        search_agent = Agent(
            role="Research Search Specialist",
            goal="Find comprehensive, credible information on research topics",
            backstory="""You are an expert researcher who excels at finding relevant, 
            high-quality sources from across the web. You understand how to craft effective 
            search queries and evaluate source credibility.""",
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.1)
        )
        
        analysis_agent = Agent(
            role="Content Analysis Expert",
            goal="Extract key insights and information from research sources",
            backstory="""You are a skilled content analyst with expertise in information 
            extraction, summarization, and identifying key themes and concepts from 
            complex documents.""",
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.1)
        )
        
        verification_agent = Agent(
            role="Fact Verification Specialist",
            goal="Verify factual claims and assess information reliability",
            backstory="""You are a meticulous fact-checker with experience in 
            cross-referencing information, validating claims, and assessing source 
            credibility across multiple domains.""",
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.1)
        )
        
        synthesis_agent = Agent(
            role="Research Synthesis Expert",
            goal="Create comprehensive, well-structured research reports",
            backstory="""You are an experienced research writer who excels at 
            synthesizing information from multiple sources into coherent, well-organized 
            reports that meet academic and professional standards.""",
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.3)
        )
        
        # Create crew
        return Crew(
            agents=[search_agent, analysis_agent, verification_agent, synthesis_agent],
            tasks=[],  # Tasks will be defined dynamically
            process=Process.sequential,
            verbose=True,
            memory=True
        )
    
    def _create_fact_check_crew(self) -> Crew:
        """Create a specialized fact-checking crew."""
        
        verification_agent = Agent(
            role="Primary Fact Checker",
            goal="Thoroughly verify factual claims using multiple credible sources",
            backstory="""You are a senior fact-checker with extensive experience in 
            information verification, source validation, and claim assessment across 
            various domains including science, politics, and current events.""",
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.1)
        )
        
        source_agent = Agent(
            role="Source Validation Expert",
            goal="Assess the credibility and reliability of information sources",
            backstory="""You specialize in evaluating source credibility, identifying 
            bias, and determining the reliability of various types of publications 
            and websites.""",
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.1)
        )
        
        return Crew(
            agents=[verification_agent, source_agent],
            tasks=[],
            process=Process.sequential,
            verbose=True
        )
    
    def _create_quick_research_crew(self) -> Crew:
        """Create a streamlined crew for quick research tasks."""
        
        quick_researcher = Agent(
            role="Quick Research Specialist",
            goal="Rapidly gather and summarize key information on topics",
            backstory="""You are an efficient researcher who can quickly identify 
            the most important information on a topic and create concise, accurate 
            summaries.""",
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.2)
        )
        
        return Crew(
            agents=[quick_researcher],
            tasks=[],
            process=Process.sequential,
            verbose=False
        )
    
    async def conduct_research(self, query: ResearchQuery) -> ResearchResult:
        """Conduct comprehensive research using the orchestrated workflow."""
        
        # Generate unique research ID
        research_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(query.query) % 10000}"
        
        # Initialize progress tracking
        progress = ResearchProgress(query=query)
        self.active_research[research_id] = progress
        
        try:
            self.logger.info(f"Starting research: {query.query} (ID: {research_id})")
            
            # Step 1: Web Search
            progress.status = "searching"
            progress.current_step = "Web search and source discovery"
            progress.progress_percentage = 10
            
            search_results = await self._perform_web_search(query)
            progress.results["search_results"] = search_results
            progress.progress_percentage = 25
            
            # Step 2: Content Analysis
            progress.status = "analyzing"
            progress.current_step = "Content analysis and information extraction"
            
            analysis_results = await self._perform_content_analysis(search_results, query)
            progress.results["analysis_results"] = analysis_results
            progress.progress_percentage = 50
            
            # Step 3: Fact Checking (if enabled)
            if query.include_fact_checking:
                progress.status = "fact_checking"
                progress.current_step = "Fact verification and claim validation"
                
                fact_check_results = await self._perform_fact_checking(analysis_results, query)
                progress.results["fact_check_results"] = fact_check_results
                progress.progress_percentage = 75
            else:
                fact_check_results = []
            
            # Step 4: Synthesis
            progress.status = "synthesizing"
            progress.current_step = "Content synthesis and report generation"
            
            synthesis_result = await self._perform_synthesis(
                search_results, analysis_results, fact_check_results, query
            )
            progress.results["synthesis_result"] = synthesis_result
            progress.progress_percentage = 90
            
            # Step 5: Finalize Results
            progress.status = "completed"
            progress.current_step = "Research completed"
            progress.progress_percentage = 100
            progress.end_time = datetime.now()
            
            # Create final result
            result = self._create_research_result(
                query, search_results, analysis_results, 
                fact_check_results, synthesis_result
            )
            
            progress.results["final_result"] = result
            
            self.logger.info(f"Research completed: {research_id}")
            return result
            
        except Exception as e:
            progress.status = "failed"
            progress.errors.append(str(e))
            progress.end_time = datetime.now()
            self.logger.error(f"Research failed: {research_id}, error: {e}")
            raise
        
        finally:
            # Clean up if needed
            if len(self.active_research) > 10:  # Keep only recent 10
                oldest_keys = sorted(self.active_research.keys())[:5]
                for key in oldest_keys:
                    del self.active_research[key]
    
    async def _perform_web_search(self, query: ResearchQuery) -> List[Dict[str, Any]]:
        """Perform web search using the web search agent."""
        
        search_query = SearchQuery(
            query=query.query,
            max_results=query.max_sources,
            language="en"
        )
        
        search_results = await self.web_search_agent.search_web(search_query)
        
        # Extract content from top results
        enriched_results = []
        for result in search_results[:query.max_sources]:
            try:
                content = await self.web_search_agent.extract_content(result.url)
                if content:
                    enriched_results.append({
                        "url": result.url,
                        "title": result.title,
                        "snippet": result.snippet,
                        "content": content,
                        "source": result.source,
                        "relevance_score": result.relevance_score,
                        "credibility_score": result.credibility_score,
                        "publish_date": result.publish_date.isoformat() if result.publish_date else None,
                        "word_count": len(content.split()) if content else 0
                    })
            except Exception as e:
                self.logger.warning(f"Failed to extract content from {result.url}: {e}")
        
        return enriched_results
    
    async def _perform_content_analysis(self, search_results: List[Dict[str, Any]], 
                                      query: ResearchQuery) -> List[Dict[str, Any]]:
        """Perform content analysis on search results."""
        
        analysis_results = []
        
        for result in search_results:
            if not result.get("content"):
                continue
            
            try:
                analysis_request = AnalysisRequest(
                    content=result["content"],
                    url=result["url"],
                    source=result.get("source"),
                    analysis_depth=query.depth,
                    focus_areas=query.focus_areas,
                    language="en"
                )
                
                analysis = await self.content_analyzer.analyze_content(analysis_request)
                
                analysis_result = {
                    "url": result["url"],
                    "title": result["title"],
                    "summary": analysis.summary,
                    "key_points": analysis.key_points,
                    "entities": analysis.entities,
                    "keywords": [{"word": word, "score": score} for word, score in analysis.keywords[:10]],
                    "topics": analysis.topics,
                    "sentiment": analysis.sentiment,
                    "readability_score": analysis.readability_score,
                    "bias_indicators": analysis.bias_indicators,
                    "factual_claims": analysis.factual_claims,
                    "credibility_score": result.get("credibility_score", 0.5),
                    "word_count": analysis.word_count,
                    "original_content": result["content"]  # Keep for synthesis
                }
                
                analysis_results.append(analysis_result)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze content from {result['url']}: {e}")
        
        return analysis_results
    
    async def _perform_fact_checking(self, analysis_results: List[Dict[str, Any]], 
                                   query: ResearchQuery) -> List[Dict[str, Any]]:
        """Perform fact checking on extracted claims."""
        
        fact_check_results = []
        
        # Collect all factual claims
        all_claims = []
        for result in analysis_results:
            for claim_text in result.get("factual_claims", []):
                claim = Claim(
                    text=claim_text,
                    source=result["url"],
                    claim_type="general"
                )
                all_claims.append((claim, result))
        
        # Verify top claims (limit to avoid timeout)
        top_claims = all_claims[:10]  # Top 10 claims
        
        for claim, source_result in top_claims:
            try:
                verification = await self.fact_checker.verify_claim(claim)
                
                fact_check_result = {
                    "claim": claim.text,
                    "source_url": source_result["url"],
                    "source_title": source_result["title"],
                    "verdict": verification.verdict,
                    "confidence_score": verification.confidence_score,
                    "supporting_sources": verification.supporting_sources,
                    "contradicting_sources": verification.contradicting_sources,
                    "verification_details": verification.verification_details,
                    "timestamp": verification.timestamp.isoformat()
                }
                
                fact_check_results.append(fact_check_result)
                
            except Exception as e:
                self.logger.warning(f"Failed to fact-check claim: {claim.text[:50]}..., error: {e}")
        
        return fact_check_results
    
    async def _perform_synthesis(self, search_results: List[Dict[str, Any]],
                               analysis_results: List[Dict[str, Any]],
                               fact_check_results: List[Dict[str, Any]],
                               query: ResearchQuery) -> Dict[str, Any]:
        """Perform content synthesis to create final report."""
        
        # Prepare source materials for synthesis
        source_materials = []
        
        for result in analysis_results:
            if result.get("credibility_score", 0) >= query.quality_threshold:
                source_material = {
                    "url": result["url"],
                    "title": result["title"],
                    "content": result["original_content"],
                    "summary": result["summary"],
                    "key_points": result["key_points"],
                    "credibility_score": result["credibility_score"],
                    "source_type": "web",
                    "author": None,  # Could be extracted if needed
                    "publication_date": None  # Could be extracted if needed
                }
                source_materials.append(source_material)
        
        if not source_materials:
            raise ValueError("No credible sources found for synthesis")
        
        # Create synthesis request
        synthesis_request = SynthesisRequest(
            topic=query.query,
            source_materials=source_materials,
            report_format=query.report_format,
            target_length=query.target_length,
            include_citations=True,
            focus_areas=query.focus_areas
        )
        
        # Perform synthesis
        synthesis_result = await self.synthesis_agent.synthesize_content(synthesis_request)
        
        return {
            "title": synthesis_result.title,
            "executive_summary": synthesis_result.executive_summary,
            "sections": [
                {
                    "title": section.title,
                    "content": section.content,
                    "section_type": section.section_type,
                    "confidence_score": section.confidence_score,
                    "source_count": len(section.sources)
                }
                for section in synthesis_result.sections
            ],
            "conclusions": synthesis_result.conclusions,
            "recommendations": synthesis_result.recommendations,
            "citations": [
                {
                    "title": citation.title,
                    "url": citation.url,
                    "author": citation.author,
                    "publication_date": citation.publication_date,
                    "credibility_score": citation.credibility_score,
                    "citation_format": citation.citation_format
                }
                for citation in synthesis_result.citations
            ],
            "metadata": synthesis_result.metadata,
            "word_count": synthesis_result.word_count,
            "quality_score": synthesis_result.quality_score
        }
    
    def _create_research_result(self, query: ResearchQuery,
                              search_results: List[Dict[str, Any]],
                              analysis_results: List[Dict[str, Any]],
                              fact_check_results: List[Dict[str, Any]],
                              synthesis_result: Dict[str, Any]) -> ResearchResult:
        """Create the final research result."""
        
        return ResearchResult(
            query=query,
            title=synthesis_result["title"],
            executive_summary=synthesis_result["executive_summary"],
            sections=synthesis_result["sections"],
            conclusions=synthesis_result["conclusions"],
            recommendations=synthesis_result["recommendations"],
            sources=search_results,
            fact_check_results=fact_check_results,
            metadata={
                "research_timestamp": datetime.now().isoformat(),
                "query_parameters": {
                    "depth": query.depth,
                    "max_sources": query.max_sources,
                    "focus_areas": query.focus_areas,
                    "report_format": query.report_format,
                    "target_length": query.target_length,
                    "include_fact_checking": query.include_fact_checking
                },
                "statistics": {
                    "sources_found": len(search_results),
                    "sources_analyzed": len(analysis_results),
                    "claims_fact_checked": len(fact_check_results),
                    "synthesis_quality": synthesis_result["quality_score"]
                }
            },
            quality_score=synthesis_result["quality_score"],
            word_count=synthesis_result["word_count"]
        )
    
    def get_research_progress(self, research_id: str) -> Optional[ResearchProgress]:
        """Get progress information for active research."""
        return self.active_research.get(research_id)
    
    def list_active_research(self) -> List[str]:
        """List all active research IDs."""
        return list(self.active_research.keys())
    
    async def cancel_research(self, research_id: str) -> bool:
        """Cancel active research."""
        if research_id in self.active_research:
            progress = self.active_research[research_id]
            progress.status = "cancelled"
            progress.end_time = datetime.now()
            return True
        return False
    
    # AMP capability handlers
    async def handle_research_request(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle research request capability."""
        query_text = parameters.get("query", "")
        depth = parameters.get("depth", "standard")
        max_sources = parameters.get("max_sources", 10)
        focus_areas = parameters.get("focus_areas", [])
        report_format = parameters.get("report_format", "academic")
        target_length = parameters.get("target_length", 1500)
        include_fact_checking = parameters.get("include_fact_checking", True)
        quality_threshold = parameters.get("quality_threshold", 0.6)
        
        if not query_text:
            return {"error": "Query is required", "result": None}
        
        query = ResearchQuery(
            query=query_text,
            depth=depth,
            max_sources=max_sources,
            focus_areas=focus_areas,
            report_format=report_format,
            target_length=target_length,
            include_fact_checking=include_fact_checking,
            quality_threshold=quality_threshold
        )
        
        try:
            result = await self.conduct_research(query)
            
            return {
                "result": {
                    "title": result.title,
                    "executive_summary": result.executive_summary,
                    "sections": result.sections,
                    "conclusions": result.conclusions,
                    "recommendations": result.recommendations,
                    "source_count": len(result.sources),
                    "fact_check_count": len(result.fact_check_results),
                    "metadata": result.metadata,
                    "quality_score": result.quality_score,
                    "word_count": result.word_count
                },
                "success": True
            }
            
        except Exception as e:
            return {"error": f"Research failed: {str(e)}", "result": None}
    
    async def handle_progress_check(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle progress check requests."""
        research_id = parameters.get("research_id", "")
        
        if not research_id:
            return {"error": "Research ID is required", "progress": None}
        
        progress = self.get_research_progress(research_id)
        
        if not progress:
            return {"error": "Research not found", "progress": None}
        
        return {
            "progress": {
                "research_id": research_id,
                "status": progress.status,
                "current_step": progress.current_step,
                "progress_percentage": progress.progress_percentage,
                "start_time": progress.start_time.isoformat(),
                "end_time": progress.end_time.isoformat() if progress.end_time else None,
                "errors": progress.errors
            },
            "success": True
        }
    
    async def start_amp_agent(self, agent_id: str = "research-orchestrator",
                             endpoint: str = "http://localhost:8000") -> AMPClient:
        """Start the AMP agent."""
        
        self.amp_client = await (
            AMPBuilder(agent_id, "Research Orchestrator")
            .with_framework("crewai")
            .with_transport(TransportType.HTTP, endpoint)
            .add_capability(
                "research-request",
                self.handle_research_request,
                "Conduct comprehensive research using coordinated agent workflows",
                "research",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "depth": {
                            "type": "string",
                            "enum": ["basic", "standard", "comprehensive"],
                            "default": "standard"
                        },
                        "max_sources": {"type": "integer", "default": 10},
                        "focus_areas": {"type": "array", "items": {"type": "string"}},
                        "report_format": {
                            "type": "string",
                            "enum": ["academic", "journalistic", "executive_summary", "technical_report"],
                            "default": "academic"
                        },
                        "target_length": {"type": "integer", "default": 1500},
                        "include_fact_checking": {"type": "boolean", "default": True},
                        "quality_threshold": {"type": "number", "default": 0.6}
                    },
                    "required": ["query"]
                },
                constraints=CapabilityConstraints(response_time_ms=120000)  # 2 minutes max
            )
            .add_capability(
                "progress-check",
                self.handle_progress_check,
                "Check progress of active research workflows",
                "monitoring",
                input_schema={
                    "type": "object",
                    "properties": {
                        "research_id": {"type": "string"}
                    },
                    "required": ["research_id"]
                }
            )
            .build()
        )
        
        return self.amp_client


async def main():
    """Main function for testing the research orchestrator."""
    logging.basicConfig(level=logging.INFO)
    
    # Create research orchestrator
    orchestrator = ResearchOrchestrator()
    
    # Start AMP agent
    client = await orchestrator.start_amp_agent()
    
    try:
        print("Research Orchestrator started. Testing research workflow...")
        
        # Test research query
        query = ResearchQuery(
            query="recent developments in renewable energy storage technologies",
            depth="standard",
            max_sources=5,
            focus_areas=["technology", "efficiency", "cost"],
            report_format="academic",
            target_length=1000,
            include_fact_checking=True
        )
        
        print(f"\nStarting research: {query.query}")
        print("=" * 60)
        
        result = await orchestrator.conduct_research(query)
        
        print(f"\nResearch Results:")
        print("=" * 60)
        print(f"Title: {result.title}")
        print(f"Word Count: {result.word_count}")
        print(f"Quality Score: {result.quality_score:.2f}")
        print(f"Sources: {len(result.sources)}")
        print(f"Fact Checks: {len(result.fact_check_results)}")
        
        print(f"\nExecutive Summary:")
        print(result.executive_summary)
        
        print(f"\nSections:")
        for section in result.sections:
            print(f"- {section['title']} (Confidence: {section['confidence_score']:.2f})")
        
        if result.conclusions:
            print(f"\nKey Conclusions:")
            for conclusion in result.conclusions[:3]:
                print(f"- {conclusion}")
        
        if result.recommendations:
            print(f"\nRecommendations:")
            for recommendation in result.recommendations[:3]:
                print(f"- {recommendation}")
        
        print("\nResearch Orchestrator is running. Press Ctrl+C to stop.")
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())