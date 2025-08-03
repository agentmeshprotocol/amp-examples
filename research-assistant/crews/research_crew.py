"""
Main Research Crew for Research Assistant Network

Defines CrewAI crew for comprehensive research workflows.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from langchain.llms import OpenAI
from langchain.tools import Tool

# Local imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from agents.web_search_agent import WebSearchAgent
from agents.content_analyzer import ContentAnalyzer
from agents.fact_checker import FactChecker
from agents.synthesis_agent import SynthesisAgent


class ResearchCrew:
    """Main research crew for comprehensive research workflows."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.ResearchCrew")
        
        # Initialize underlying agents
        self.web_search_agent = WebSearchAgent(config.get("web_search", {}))
        self.content_analyzer = ContentAnalyzer(config.get("content_analysis", {}))
        self.fact_checker = FactChecker(config.get("fact_checking", {}))
        self.synthesis_agent = SynthesisAgent(config.get("synthesis", {}))
        
        # Create crew
        self.crew = self._create_crew()
    
    def _create_crew(self) -> Crew:
        """Create the comprehensive research crew."""
        
        # Define agents
        search_agent = Agent(
            role="Senior Research Search Specialist",
            goal="Find the most comprehensive and credible information sources on research topics",
            backstory="""You are a world-class research librarian and information scientist 
            with 15+ years of experience in academic and professional research. You have an 
            exceptional ability to craft precise search queries, identify authoritative sources, 
            and distinguish between reliable and unreliable information. Your expertise spans 
            multiple domains including science, technology, business, and current affairs.""",
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.1),
            tools=self._create_search_tools()
        )
        
        analysis_agent = Agent(
            role="Expert Content Analysis Researcher",
            goal="Extract comprehensive insights, key information, and themes from research materials",
            backstory="""You are a distinguished content analyst and information extraction 
            expert with extensive experience in natural language processing and academic research. 
            You excel at identifying key concepts, extracting meaningful insights, detecting bias, 
            and summarizing complex information while maintaining accuracy and nuance. Your 
            analytical skills have been honed through years of working with diverse content 
            types and research domains.""",
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.1),
            tools=self._create_analysis_tools()
        )
        
        verification_agent = Agent(
            role="Senior Fact Verification Specialist",
            goal="Rigorously verify factual claims and assess information reliability using multiple credible sources",
            backstory="""You are a seasoned fact-checker and verification expert with 20+ years 
            of experience in journalism, academic research, and information validation. You have 
            worked with major news organizations and research institutions, developing expertise 
            in cross-referencing sources, identifying misinformation, and assessing credibility. 
            Your meticulous approach and deep understanding of verification methodologies make 
            you the gold standard for factual accuracy.""",
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.1),
            tools=self._create_verification_tools()
        )
        
        synthesis_agent = Agent(
            role="Master Research Synthesis Expert",
            goal="Create exceptional, well-structured research reports that synthesize information from multiple sources",
            backstory="""You are a renowned research writer and synthesis expert with a 
            distinguished career in academic publishing and professional research. You have 
            authored numerous peer-reviewed papers and comprehensive reports, with expertise 
            in organizing complex information, maintaining narrative coherence, and creating 
            compelling, evidence-based arguments. Your ability to weave together insights 
            from multiple sources while maintaining academic rigor is unparalleled.""",
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.3),
            tools=self._create_synthesis_tools()
        )
        
        # Create crew with sequential process
        return Crew(
            agents=[search_agent, analysis_agent, verification_agent, synthesis_agent],
            tasks=[],  # Tasks will be created dynamically
            process=Process.sequential,
            verbose=True,
            memory=True,
            max_execution_time=3600  # 1 hour timeout
        )
    
    def _create_search_tools(self) -> List[Tool]:
        """Create tools for the search agent."""
        tools = []
        
        # Web search tool
        async def web_search_tool(query: str) -> str:
            """Search the web for information on a topic."""
            try:
                from agents.web_search_agent import SearchQuery
                search_query = SearchQuery(query=query, max_results=10)
                results = await self.web_search_agent.search_web(search_query)
                
                formatted_results = []
                for result in results:
                    formatted_results.append(
                        f"Title: {result.title}\n"
                        f"URL: {result.url}\n"
                        f"Snippet: {result.snippet}\n"
                        f"Credibility: {result.credibility_score:.2f}\n"
                        f"---"
                    )
                return "\n".join(formatted_results)
            except Exception as e:
                return f"Search failed: {str(e)}"
        
        tools.append(Tool(
            name="web_search",
            description="Search the web for information on a specific topic",
            func=web_search_tool
        ))
        
        # Content extraction tool
        async def extract_content_tool(url: str) -> str:
            """Extract full content from a web page."""
            try:
                content = await self.web_search_agent.extract_content(url)
                return content if content else "Failed to extract content"
            except Exception as e:
                return f"Content extraction failed: {str(e)}"
        
        tools.append(Tool(
            name="extract_content",
            description="Extract full text content from a web page URL",
            func=extract_content_tool
        ))
        
        return tools
    
    def _create_analysis_tools(self) -> List[Tool]:
        """Create tools for the analysis agent."""
        tools = []
        
        # Content analysis tool
        async def analyze_content_tool(content: str) -> str:
            """Analyze content and extract key information."""
            try:
                from agents.content_analyzer import AnalysisRequest
                request = AnalysisRequest(
                    content=content,
                    analysis_depth="comprehensive"
                )
                analysis = await self.content_analyzer.analyze_content(request)
                
                result = f"""
Analysis Results:
Summary: {analysis.summary}

Key Points:
{chr(10).join(f"- {point}" for point in analysis.key_points[:5])}

Top Keywords:
{chr(10).join(f"- {word}: {score:.3f}" for word, score in analysis.keywords[:10])}

Entities Found:
{chr(10).join(f"- {ent['text']} ({ent['label']})" for ent in analysis.entities[:10])}

Sentiment: {analysis.sentiment}
Readability Score: {analysis.readability_score:.1f}

Bias Indicators: {', '.join(analysis.bias_indicators) if analysis.bias_indicators else 'None detected'}

Factual Claims:
{chr(10).join(f"- {claim}" for claim in analysis.factual_claims[:3])}
"""
                return result
            except Exception as e:
                return f"Analysis failed: {str(e)}"
        
        tools.append(Tool(
            name="analyze_content",
            description="Analyze text content to extract key information, entities, and insights",
            func=analyze_content_tool
        ))
        
        return tools
    
    def _create_verification_tools(self) -> List[Tool]:
        """Create tools for the verification agent."""
        tools = []
        
        # Fact verification tool
        async def verify_claim_tool(claim: str) -> str:
            """Verify a factual claim using multiple sources."""
            try:
                from agents.fact_checker import Claim
                claim_obj = Claim(text=claim, claim_type="general")
                result = await self.fact_checker.verify_claim(claim_obj)
                
                verification_result = f"""
Fact Check Result for: "{claim}"

Verdict: {result.verdict}
Confidence: {result.confidence_score:.2f}

Supporting Sources ({len(result.supporting_sources)}):
{chr(10).join(f"- {src.get('title', 'Unknown')} (Credibility: {src.get('credibility_score', 0):.2f})" for src in result.supporting_sources[:3])}

Contradicting Sources ({len(result.contradicting_sources)}):
{chr(10).join(f"- {src.get('title', 'Unknown')} (Credibility: {src.get('credibility_score', 0):.2f})" for src in result.contradicting_sources[:3])}

Verification Details:
- Sources checked: {result.verification_details.get('sources_checked', 0)}
- Search queries used: {len(result.verification_details.get('search_queries', []))}
"""
                return verification_result
            except Exception as e:
                return f"Verification failed: {str(e)}"
        
        tools.append(Tool(
            name="verify_claim",
            description="Verify a specific factual claim using multiple credible sources",
            func=verify_claim_tool
        ))
        
        # Source credibility tool
        async def assess_credibility_tool(url: str) -> str:
            """Assess the credibility of a source."""
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc.lower()
                
                # Use fact checker's credibility assessment
                if domain.startswith('www.'):
                    domain = domain[4:]
                
                credibility_score = self.fact_checker._calculate_domain_credibility(domain)
                
                if domain in self.fact_checker.credible_sources:
                    source_info = self.fact_checker.credible_sources[domain]
                    return f"""
Source Credibility Assessment for: {url}

Domain: {domain}
Credibility Score: {source_info['credibility']:.2f}
Source Type: {source_info['type']}
Bias: {source_info['bias']}
Known Source: Yes
"""
                else:
                    return f"""
Source Credibility Assessment for: {url}

Domain: {domain}
Credibility Score: {credibility_score:.2f}
Known Source: No
Assessment: Calculated based on domain characteristics
"""
            except Exception as e:
                return f"Credibility assessment failed: {str(e)}"
        
        tools.append(Tool(
            name="assess_credibility",
            description="Assess the credibility and reliability of an information source",
            func=assess_credibility_tool
        ))
        
        return tools
    
    def _create_synthesis_tools(self) -> List[Tool]:
        """Create tools for the synthesis agent."""
        tools = []
        
        # Content synthesis tool
        async def synthesize_content_tool(topic: str, sources_data: str) -> str:
            """Synthesize content from multiple sources into a coherent report."""
            try:
                # Parse sources data (simplified for this example)
                # In practice, you'd have a more sophisticated parsing mechanism
                source_materials = [
                    {
                        "url": "example.com",
                        "title": "Synthesized Content",
                        "content": sources_data,
                        "credibility_score": 0.8,
                        "source_type": "web"
                    }
                ]
                
                from agents.synthesis_agent import SynthesisRequest
                request = SynthesisRequest(
                    topic=topic,
                    source_materials=source_materials,
                    report_format="academic",
                    target_length=1500
                )
                
                result = await self.synthesis_agent.synthesize_content(request)
                
                synthesis_result = f"""
Synthesis Report: {result.title}

Executive Summary:
{result.executive_summary}

Main Sections:
{chr(10).join(f"- {section.title}: {len(section.content.split())} words (Confidence: {section.confidence_score:.2f})" for section in result.sections)}

Key Conclusions:
{chr(10).join(f"- {conclusion}" for conclusion in result.conclusions[:3])}

Recommendations:
{chr(10).join(f"- {recommendation}" for recommendation in result.recommendations[:3])}

Total Word Count: {result.word_count}
Quality Score: {result.quality_score:.2f}
"""
                return synthesis_result
            except Exception as e:
                return f"Synthesis failed: {str(e)}"
        
        tools.append(Tool(
            name="synthesize_content",
            description="Synthesize information from multiple sources into a comprehensive report",
            func=synthesize_content_tool
        ))
        
        return tools
    
    async def execute_research(self, query: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a comprehensive research workflow."""
        
        # Create dynamic tasks based on requirements
        tasks = self._create_research_tasks(query, requirements)
        
        # Update crew with new tasks
        self.crew.tasks = tasks
        
        try:
            # Execute the crew
            result = await asyncio.to_thread(self.crew.kickoff)
            
            return {
                "success": True,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "requirements": requirements
            }
            
        except Exception as e:
            self.logger.error(f"Research execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "query": query
            }
    
    def _create_research_tasks(self, query: str, requirements: Dict[str, Any]) -> List[Task]:
        """Create research tasks based on query and requirements."""
        
        tasks = []
        
        # Task 1: Information Discovery
        search_task = Task(
            description=f"""
            Conduct comprehensive web search on: "{query}"
            
            Requirements:
            - Find {requirements.get('max_sources', 10)} high-quality sources
            - Focus on credible, authoritative sources
            - Extract full content from top sources
            - Assess credibility of each source
            
            Focus areas: {', '.join(requirements.get('focus_areas', []))}
            
            Deliver a comprehensive list of sources with content and credibility assessments.
            """,
            agent=self.crew.agents[0],  # Search agent
            expected_output="Detailed list of sources with content extracts and credibility scores"
        )
        tasks.append(search_task)
        
        # Task 2: Content Analysis
        analysis_task = Task(
            description=f"""
            Analyze all collected content for: "{query}"
            
            Requirements:
            - Extract key insights and themes
            - Identify important entities and concepts
            - Summarize main findings
            - Detect potential bias or limitations
            - Extract factual claims for verification
            
            Analysis depth: {requirements.get('depth', 'standard')}
            
            Provide comprehensive analysis with key points, entities, and factual claims.
            """,
            agent=self.crew.agents[1],  # Analysis agent
            expected_output="Comprehensive content analysis with key insights and factual claims"
        )
        tasks.append(analysis_task)
        
        # Task 3: Fact Verification (if enabled)
        if requirements.get('include_fact_checking', True):
            verification_task = Task(
                description=f"""
                Verify key factual claims about: "{query}"
                
                Requirements:
                - Verify the most important factual claims identified in analysis
                - Cross-reference with multiple credible sources
                - Assess claim accuracy and confidence levels
                - Document supporting and contradicting evidence
                
                Quality threshold: {requirements.get('quality_threshold', 0.6)}
                
                Provide detailed fact-check results with evidence and confidence scores.
                """,
                agent=self.crew.agents[2],  # Verification agent
                expected_output="Detailed fact-check results with verification evidence"
            )
            tasks.append(verification_task)
        
        # Task 4: Content Synthesis
        synthesis_task = Task(
            description=f"""
            Create a comprehensive research report on: "{query}"
            
            Requirements:
            - Synthesize all verified information into a coherent report
            - Follow {requirements.get('report_format', 'academic')} format
            - Target length: {requirements.get('target_length', 1500)} words
            - Include proper citations and references
            - Organize into clear sections with conclusions and recommendations
            
            Focus areas: {', '.join(requirements.get('focus_areas', []))}
            
            Deliver a polished, comprehensive research report with citations.
            """,
            agent=self.crew.agents[-1],  # Synthesis agent
            expected_output="Complete research report with executive summary, sections, conclusions, and citations"
        )
        tasks.append(synthesis_task)
        
        return tasks


async def main():
    """Test the research crew."""
    logging.basicConfig(level=logging.INFO)
    
    # Create research crew
    crew = ResearchCrew()
    
    print("Research Crew initialized. Testing research workflow...")
    
    # Test query
    query = "renewable energy storage breakthrough technologies 2024"
    requirements = {
        "depth": "standard",
        "max_sources": 5,
        "focus_areas": ["technology", "efficiency", "commercialization"],
        "report_format": "academic",
        "target_length": 1200,
        "include_fact_checking": True,
        "quality_threshold": 0.7
    }
    
    print(f"\nExecuting research on: {query}")
    print("=" * 60)
    
    try:
        result = await crew.execute_research(query, requirements)
        
        if result["success"]:
            print("Research completed successfully!")
            print(f"Result: {result['result']}")
        else:
            print(f"Research failed: {result['error']}")
    
    except Exception as e:
        print(f"Error during research: {e}")


if __name__ == "__main__":
    asyncio.run(main())