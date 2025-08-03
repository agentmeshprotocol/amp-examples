"""
Fact Check Crew for Research Assistant Network

Specialized CrewAI crew focused on fact verification and claim validation.
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
from agents.fact_checker import FactChecker, Claim
from agents.web_search_agent import WebSearchAgent


class FactCheckCrew:
    """Specialized crew for comprehensive fact-checking workflows."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.FactCheckCrew")
        
        # Initialize underlying agents
        self.fact_checker = FactChecker(config.get("fact_checking", {}))
        self.web_search_agent = WebSearchAgent(config.get("web_search", {}))
        
        # Create crew
        self.crew = self._create_crew()
    
    def _create_crew(self) -> Crew:
        """Create the specialized fact-checking crew."""
        
        # Primary Fact Verification Agent
        primary_verifier = Agent(
            role="Lead Fact Verification Specialist",
            goal="Conduct thorough, accurate fact verification using multiple authoritative sources",
            backstory="""You are a world-renowned fact-checker with over 15 years of experience 
            working with major news organizations, academic institutions, and fact-checking 
            organizations like Snopes and PolitiFact. You have developed rigorous methodologies 
            for claim verification and have an exceptional track record for accuracy. Your 
            expertise includes identifying reliable sources, cross-referencing information, 
            and detecting misinformation across various domains including science, politics, 
            health, and current events.""",
            verbose=True,
            allow_delegation=True,
            llm=OpenAI(temperature=0.05),  # Very low temperature for precision
            tools=self._create_verification_tools()
        )
        
        # Source Validation Agent
        source_validator = Agent(
            role="Source Credibility and Validation Expert",
            goal="Assess source reliability, credibility, and potential bias to ensure fact-checking accuracy",
            backstory="""You are a distinguished information scientist and media literacy expert 
            with deep expertise in evaluating source credibility. You have worked with universities, 
            libraries, and research institutions to develop source evaluation frameworks. Your 
            specialty is identifying publication bias, financial conflicts of interest, and 
            methodological issues that could affect information reliability. You understand 
            the nuances of different types of sources from peer-reviewed journals to news outlets.""",
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.1),
            tools=self._create_source_validation_tools()
        )
        
        # Claims Analysis Agent
        claims_analyst = Agent(
            role="Claims Analysis and Decomposition Specialist",
            goal="Break down complex claims into verifiable components and identify key verification points",
            backstory="""You are an expert in logic, argumentation, and claims analysis with 
            a background in philosophy and critical thinking. You excel at dissecting complex 
            statements, identifying implicit assumptions, and breaking down multi-faceted claims 
            into discrete, verifiable elements. Your analytical skills help ensure that fact-checking 
            is comprehensive and addresses all aspects of a claim systematically.""",
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.2),
            tools=self._create_claims_analysis_tools()
        )
        
        # Evidence Synthesis Agent
        evidence_synthesizer = Agent(
            role="Evidence Synthesis and Verdict Specialist",
            goal="Synthesize verification evidence to reach accurate, well-supported conclusions",
            backstory="""You are a senior research analyst with expertise in evidence evaluation, 
            meta-analysis, and scientific reasoning. You have worked with courts, research 
            institutions, and policy organizations to evaluate complex evidence. Your strength 
            lies in weighing conflicting evidence, assessing confidence levels, and reaching 
            balanced conclusions that accurately reflect the available evidence while acknowledging 
            limitations and uncertainties.""",
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.15),
            tools=self._create_evidence_synthesis_tools()
        )
        
        # Create crew with hierarchical process for complex fact-checking
        return Crew(
            agents=[primary_verifier, source_validator, claims_analyst, evidence_synthesizer],
            tasks=[],  # Tasks will be created dynamically
            process=Process.hierarchical,
            manager_llm=OpenAI(temperature=0.1),
            verbose=True,
            memory=True,
            max_execution_time=1800  # 30 minutes timeout
        )
    
    def _create_verification_tools(self) -> List[Tool]:
        """Create tools for the primary verification agent."""
        tools = []
        
        # Claim verification tool
        async def verify_specific_claim(claim: str) -> str:
            """Verify a specific claim using comprehensive fact-checking."""
            try:
                claim_obj = Claim(text=claim, claim_type="general")
                result = await self.fact_checker.verify_claim(claim_obj)
                
                return f"""
FACT CHECK RESULT:
Claim: "{claim}"
Verdict: {result.verdict.upper()}
Confidence: {result.confidence_score:.2f}

SUPPORTING EVIDENCE ({len(result.supporting_sources)} sources):
{chr(10).join(f"✓ {src.get('title', 'Unknown')[:80]}... (Credibility: {src.get('credibility_score', 0):.2f})" for src in result.supporting_sources[:5])}

CONTRADICTING EVIDENCE ({len(result.contradicting_sources)} sources):
{chr(10).join(f"✗ {src.get('title', 'Unknown')[:80]}... (Credibility: {src.get('credibility_score', 0):.2f})" for src in result.contradicting_sources[:5])}

VERIFICATION DETAILS:
- Total sources checked: {result.verification_details.get('sources_checked', 0)}
- Search queries used: {len(result.verification_details.get('search_queries', []))}
- Verification method: {result.verification_details.get('verification_method', 'Unknown')}

RELEVANT EXCERPTS:
{chr(10).join(f"• {excerpt[:200]}..." for src in result.supporting_sources[:2] for excerpt in src.get('relevant_excerpts', [])[:1])}
"""
            except Exception as e:
                return f"Verification failed: {str(e)}"
        
        tools.append(Tool(
            name="verify_claim",
            description="Verify a specific factual claim using comprehensive source checking",
            func=verify_specific_claim
        ))
        
        # Cross-reference verification tool
        async def cross_reference_claim(claim: str, additional_context: str = "") -> str:
            """Cross-reference a claim against multiple source types."""
            try:
                from agents.web_search_agent import SearchQuery
                
                # Generate multiple search queries for cross-referencing
                base_query = claim
                context_query = f"{claim} {additional_context}" if additional_context else claim
                fact_check_query = f"fact check {claim}"
                study_query = f"study research {claim}"
                
                all_results = []
                for query in [base_query, context_query, fact_check_query, study_query]:
                    search_query = SearchQuery(query=query, max_results=5)
                    results = await self.web_search_agent.search_web(search_query)
                    all_results.extend(results)
                
                # Group by source type
                source_types = {"academic": [], "news": [], "government": [], "fact_check": [], "other": []}
                
                for result in all_results:
                    url_lower = result.url.lower()
                    if any(domain in url_lower for domain in ['.edu', 'pubmed', 'arxiv', 'nature.com', 'science.org']):
                        source_types["academic"].append(result)
                    elif any(domain in url_lower for domain in ['.gov', 'cdc.gov', 'who.int']):
                        source_types["government"].append(result)
                    elif any(domain in url_lower for domain in ['snopes', 'factcheck', 'politifact']):
                        source_types["fact_check"].append(result)
                    elif any(domain in url_lower for domain in ['reuters', 'ap.org', 'bbc', 'npr']):
                        source_types["news"].append(result)
                    else:
                        source_types["other"].append(result)
                
                cross_ref_report = f"""
CROSS-REFERENCE ANALYSIS for: "{claim}"

ACADEMIC SOURCES ({len(source_types['academic'])}):
{chr(10).join(f"• {r.title[:60]}... (Credibility: {r.credibility_score:.2f})" for r in source_types['academic'][:3])}

GOVERNMENT SOURCES ({len(source_types['government'])}):
{chr(10).join(f"• {r.title[:60]}... (Credibility: {r.credibility_score:.2f})" for r in source_types['government'][:3])}

FACT-CHECK SOURCES ({len(source_types['fact_check'])}):
{chr(10).join(f"• {r.title[:60]}... (Credibility: {r.credibility_score:.2f})" for r in source_types['fact_check'][:3])}

NEWS SOURCES ({len(source_types['news'])}):
{chr(10).join(f"• {r.title[:60]}... (Credibility: {r.credibility_score:.2f})" for r in source_types['news'][:3])}

CROSS-REFERENCE SUMMARY:
- Total unique sources found: {len(all_results)}
- Source diversity score: {len([k for k, v in source_types.items() if v]) / 5:.2f}
- Average credibility: {sum(r.credibility_score for r in all_results) / len(all_results):.2f}
"""
                return cross_ref_report
                
            except Exception as e:
                return f"Cross-reference failed: {str(e)}"
        
        tools.append(Tool(
            name="cross_reference",
            description="Cross-reference a claim against multiple types of sources (academic, government, news, fact-check)",
            func=cross_reference_claim
        ))
        
        return tools
    
    def _create_source_validation_tools(self) -> List[Tool]:
        """Create tools for source validation."""
        tools = []
        
        # Source credibility assessment tool
        async def assess_source_credibility(url: str) -> str:
            """Assess comprehensive credibility of a source."""
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc.lower()
                
                if domain.startswith('www.'):
                    domain = domain[4:]
                
                # Get detailed credibility assessment
                if domain in self.fact_checker.credible_sources:
                    source_info = self.fact_checker.credible_sources[domain]
                    credibility_details = f"""
CREDIBILITY ASSESSMENT: {url}

DOMAIN: {domain}
CREDIBILITY SCORE: {source_info['credibility']:.2f} / 1.0
SOURCE TYPE: {source_info['type'].title()}
POLITICAL BIAS: {source_info['bias'].title()}
KNOWN SOURCE: ✓ Yes (in verified database)

RELIABILITY FACTORS:
✓ Established publication with verification standards
✓ Professional editorial oversight
✓ Transparent correction policies
✓ Verifiable contact information and staff

RECOMMENDATION: This is a {source_info['type']} source with {source_info['credibility']:.0%} credibility rating.
"""
                else:
                    calculated_score = self.fact_checker._calculate_domain_credibility(domain)
                    credibility_details = f"""
CREDIBILITY ASSESSMENT: {url}

DOMAIN: {domain}
CREDIBILITY SCORE: {calculated_score:.2f} / 1.0
KNOWN SOURCE: ✗ Not in verified database
ASSESSMENT: Calculated based on domain characteristics

DOMAIN ANALYSIS:
- TLD Analysis: {self._analyze_tld(domain)}
- Domain Age: Unknown
- Publication Type: Inferred from domain structure

RECOMMENDATION: Use with caution - verify claims independently.
"""
                
                return credibility_details
                
            except Exception as e:
                return f"Credibility assessment failed: {str(e)}"
        
        tools.append(Tool(
            name="assess_credibility",
            description="Conduct comprehensive credibility assessment of information sources",
            func=assess_source_credibility
        ))
        
        # Bias detection tool
        async def detect_source_bias(content: str, url: str = "") -> str:
            """Detect potential bias in source content."""
            try:
                bias_indicators = []
                content_lower = content.lower()
                
                # Emotional language indicators
                emotional_words = ['shocking', 'outrageous', 'devastating', 'incredible', 'unbelievable']
                found_emotional = [word for word in emotional_words if word in content_lower]
                if found_emotional:
                    bias_indicators.append(f"Emotional language: {', '.join(found_emotional)}")
                
                # Opinion vs fact indicators
                opinion_phrases = ['i believe', 'i think', 'clearly', 'obviously', 'undoubtedly']
                found_opinions = [phrase for phrase in opinion_phrases if phrase in content_lower]
                if found_opinions:
                    bias_indicators.append(f"Opinion indicators: {', '.join(found_opinions)}")
                
                # Absolute statements
                absolute_words = ['always', 'never', 'all', 'none', 'completely', 'totally']
                found_absolutes = [word for word in absolute_words if word in content_lower]
                if found_absolutes:
                    bias_indicators.append(f"Absolute statements: {', '.join(found_absolutes)}")
                
                # Source attribution analysis
                source_phrases = ['according to', 'study shows', 'research indicates', 'data reveals']
                has_attribution = any(phrase in content_lower for phrase in source_phrases)
                
                bias_report = f"""
BIAS ANALYSIS for: {url if url else 'Content'}

POTENTIAL BIAS INDICATORS:
{chr(10).join(f"⚠ {indicator}" for indicator in bias_indicators) if bias_indicators else "✓ No major bias indicators detected"}

SOURCE ATTRIBUTION:
{'✓ Contains source attribution' if has_attribution else '⚠ Limited source attribution'}

OBJECTIVITY SCORE: {max(0, 1.0 - len(bias_indicators) * 0.2):.2f} / 1.0

RECOMMENDATION: {'Use with caution due to bias indicators' if bias_indicators else 'Content appears relatively objective'}
"""
                return bias_report
                
            except Exception as e:
                return f"Bias detection failed: {str(e)}"
        
        tools.append(Tool(
            name="detect_bias",
            description="Analyze content for potential bias indicators and objectivity",
            func=detect_source_bias
        ))
        
        return tools
    
    def _create_claims_analysis_tools(self) -> List[Tool]:
        """Create tools for claims analysis."""
        tools = []
        
        # Claim decomposition tool
        def decompose_claim(claim: str) -> str:
            """Break down complex claims into verifiable components."""
            try:
                # Identify claim components
                components = []
                
                # Statistical claims
                import re
                numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|percent|million|billion|thousand)?\b', claim)
                if numbers:
                    components.append(f"Numerical/Statistical: {', '.join(numbers)}")
                
                # Temporal claims
                dates = re.findall(r'\b(19|20)\d{2}\b|\b(?:since|until|by|in)\s+\d{4}\b', claim)
                if dates:
                    components.append(f"Temporal: {', '.join(dates)}")
                
                # Causal claims
                causal_indicators = ['causes', 'leads to', 'results in', 'due to', 'because of']
                found_causal = [indicator for indicator in causal_indicators if indicator in claim.lower()]
                if found_causal:
                    components.append(f"Causal: {', '.join(found_causal)}")
                
                # Comparative claims
                comparative_words = ['more', 'less', 'better', 'worse', 'higher', 'lower', 'compared to']
                found_comparative = [word for word in comparative_words if word in claim.lower()]
                if found_comparative:
                    components.append(f"Comparative: {', '.join(found_comparative)}")
                
                decomposition = f"""
CLAIM DECOMPOSITION: "{claim}"

IDENTIFIED COMPONENTS:
{chr(10).join(f"• {component}" for component in components) if components else "• Single, direct factual claim"}

VERIFICATION STRATEGY:
{self._suggest_verification_strategy(claim, components)}

COMPLEXITY LEVEL: {self._assess_claim_complexity(claim, components)}
"""
                return decomposition
                
            except Exception as e:
                return f"Claim decomposition failed: {str(e)}"
        
        tools.append(Tool(
            name="decompose_claim",
            description="Break down complex claims into verifiable components",
            func=decompose_claim
        ))
        
        return tools
    
    def _create_evidence_synthesis_tools(self) -> List[Tool]:
        """Create tools for evidence synthesis."""
        tools = []
        
        # Evidence evaluation tool
        def evaluate_evidence(evidence_summary: str) -> str:
            """Evaluate and synthesize evidence to reach conclusions."""
            try:
                evidence_evaluation = f"""
EVIDENCE EVALUATION

EVIDENCE STRENGTH ANALYSIS:
{self._analyze_evidence_strength(evidence_summary)}

CONFIDENCE ASSESSMENT:
{self._assess_confidence_level(evidence_summary)}

FINAL VERDICT RECOMMENDATION:
{self._recommend_verdict(evidence_summary)}

LIMITATIONS AND CAVEATS:
{self._identify_limitations(evidence_summary)}
"""
                return evidence_evaluation
                
            except Exception as e:
                return f"Evidence evaluation failed: {str(e)}"
        
        tools.append(Tool(
            name="evaluate_evidence",
            description="Evaluate evidence strength and synthesize to reach fact-checking conclusions",
            func=evaluate_evidence
        ))
        
        return tools
    
    def _analyze_tld(self, domain: str) -> str:
        """Analyze top-level domain for credibility indicators."""
        if domain.endswith('.gov'):
            return "Government domain (high credibility)"
        elif domain.endswith('.edu'):
            return "Educational institution (high credibility)"
        elif domain.endswith('.org'):
            return "Organization domain (varies)"
        elif domain.endswith('.com'):
            return "Commercial domain (varies)"
        else:
            return "Other TLD"
    
    def _suggest_verification_strategy(self, claim: str, components: List[str]) -> str:
        """Suggest verification strategy based on claim components."""
        strategies = []
        
        if any("Numerical" in comp for comp in components):
            strategies.append("Verify statistics with original data sources")
        if any("Temporal" in comp for comp in components):
            strategies.append("Check historical records and timelines")
        if any("Causal" in comp for comp in components):
            strategies.append("Look for peer-reviewed studies on causation")
        if any("Comparative" in comp for comp in components):
            strategies.append("Find comparative data from reliable sources")
        
        if not strategies:
            strategies.append("Direct source verification with authoritative references")
        
        return "; ".join(strategies)
    
    def _assess_claim_complexity(self, claim: str, components: List[str]) -> str:
        """Assess the complexity level of a claim."""
        if len(components) >= 3:
            return "High (multi-faceted claim requiring comprehensive verification)"
        elif len(components) == 2:
            return "Medium (compound claim with multiple verification points)"
        else:
            return "Low (single-component claim)"
    
    def _analyze_evidence_strength(self, evidence: str) -> str:
        """Analyze the strength of evidence."""
        # Simplified analysis based on keywords in evidence summary
        strength_indicators = {
            "strong": ["peer-reviewed", "multiple studies", "government data", "consensus"],
            "moderate": ["reputable source", "consistent reporting", "expert opinion"],
            "weak": ["single source", "anecdotal", "unverified", "conflicting"]
        }
        
        evidence_lower = evidence.lower()
        
        for strength, indicators in strength_indicators.items():
            if any(indicator in evidence_lower for indicator in indicators):
                return f"{strength.title()} evidence quality"
        
        return "Evidence quality unclear"
    
    def _assess_confidence_level(self, evidence: str) -> str:
        """Assess confidence level based on evidence."""
        evidence_lower = evidence.lower()
        
        if "contradicting" in evidence_lower and "supporting" in evidence_lower:
            return "Medium confidence (mixed evidence)"
        elif "multiple" in evidence_lower and "credible" in evidence_lower:
            return "High confidence (strong supporting evidence)"
        elif "limited" in evidence_lower or "insufficient" in evidence_lower:
            return "Low confidence (insufficient evidence)"
        else:
            return "Moderate confidence"
    
    def _recommend_verdict(self, evidence: str) -> str:
        """Recommend final verdict based on evidence."""
        evidence_lower = evidence.lower()
        
        if "strong support" in evidence_lower or "verified" in evidence_lower:
            return "SUPPORTED - Evidence strongly supports the claim"
        elif "contradicted" in evidence_lower or "false" in evidence_lower:
            return "CONTRADICTED - Evidence contradicts the claim"
        elif "mixed" in evidence_lower or "conflicting" in evidence_lower:
            return "MIXED - Evidence shows conflicting information"
        else:
            return "UNVERIFIED - Insufficient evidence to determine accuracy"
    
    def _identify_limitations(self, evidence: str) -> str:
        """Identify limitations in the evidence or verification process."""
        limitations = []
        evidence_lower = evidence.lower()
        
        if "limited sources" in evidence_lower:
            limitations.append("Limited number of sources available")
        if "recent" in evidence_lower or "new" in evidence_lower:
            limitations.append("Topic may be too recent for comprehensive verification")
        if "technical" in evidence_lower or "specialized" in evidence_lower:
            limitations.append("Requires specialized expertise for full evaluation")
        
        return "; ".join(limitations) if limitations else "No major limitations identified"
    
    async def execute_fact_check(self, claims: List[str], context: str = "") -> Dict[str, Any]:
        """Execute comprehensive fact-checking workflow."""
        
        try:
            # Create dynamic tasks for fact-checking
            tasks = self._create_fact_check_tasks(claims, context)
            
            # Update crew with new tasks
            self.crew.tasks = tasks
            
            # Execute the crew
            result = await asyncio.to_thread(self.crew.kickoff)
            
            return {
                "success": True,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "claims_checked": len(claims),
                "context": context
            }
            
        except Exception as e:
            self.logger.error(f"Fact-check execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "claims": claims
            }
    
    def _create_fact_check_tasks(self, claims: List[str], context: str) -> List[Task]:
        """Create fact-checking tasks."""
        
        tasks = []
        
        # Task 1: Claims Analysis and Decomposition
        analysis_task = Task(
            description=f"""
            Analyze and decompose the following claims for systematic verification:
            
            Claims to verify:
            {chr(10).join(f"- {claim}" for claim in claims)}
            
            Context: {context}
            
            For each claim:
            1. Break down into verifiable components
            2. Identify claim type (statistical, causal, temporal, etc.)
            3. Suggest optimal verification strategy
            4. Assess complexity level
            
            Provide detailed analysis for systematic fact-checking.
            """,
            agent=self.crew.agents[2],  # Claims analyst
            expected_output="Detailed claim analysis with verification strategies"
        )
        tasks.append(analysis_task)
        
        # Task 2: Source Validation and Credibility Assessment
        validation_task = Task(
            description=f"""
            Identify and validate sources for fact-checking these claims:
            
            Claims: {', '.join(claims[:3])}...
            
            Requirements:
            1. Find authoritative sources for each claim type
            2. Assess source credibility and potential bias
            3. Identify the most reliable sources for verification
            4. Flag any problematic sources or bias indicators
            
            Focus on academic, government, and established fact-checking sources.
            """,
            agent=self.crew.agents[1],  # Source validator
            expected_output="Validated source list with credibility assessments"
        )
        tasks.append(validation_task)
        
        # Task 3: Primary Fact Verification
        verification_task = Task(
            description=f"""
            Conduct comprehensive fact verification for all claims:
            
            Claims to verify:
            {chr(10).join(f"- {claim}" for claim in claims)}
            
            For each claim:
            1. Verify against multiple credible sources
            2. Cross-reference with different source types
            3. Document supporting and contradicting evidence
            4. Assess confidence levels
            
            Use the validated sources and verification strategies from previous analysis.
            """,
            agent=self.crew.agents[0],  # Primary verifier
            expected_output="Complete verification results with evidence documentation"
        )
        tasks.append(verification_task)
        
        # Task 4: Evidence Synthesis and Final Verdict
        synthesis_task = Task(
            description=f"""
            Synthesize all verification evidence to reach final conclusions:
            
            Requirements:
            1. Evaluate strength and quality of evidence for each claim
            2. Weigh supporting vs. contradicting evidence
            3. Assign confidence levels and final verdicts
            4. Identify any limitations or caveats
            5. Provide clear, evidence-based conclusions
            
            Deliver comprehensive fact-check report with final verdicts.
            """,
            agent=self.crew.agents[3],  # Evidence synthesizer
            expected_output="Final fact-check report with verdicts and confidence levels"
        )
        tasks.append(synthesis_task)
        
        return tasks


async def main():
    """Test the fact-check crew."""
    logging.basicConfig(level=logging.INFO)
    
    # Create fact-check crew
    crew = FactCheckCrew()
    
    print("Fact Check Crew initialized. Testing fact-checking workflow...")
    
    # Test claims
    claims = [
        "COVID-19 vaccines are 95% effective in preventing severe illness",
        "Solar panels lose 50% efficiency after 10 years",
        "Electric vehicles produce 50% less CO2 than gasoline cars over their lifetime"
    ]
    
    context = "Claims related to recent technology and health developments"
    
    print(f"\nFact-checking {len(claims)} claims...")
    print("=" * 60)
    
    try:
        result = await crew.execute_fact_check(claims, context)
        
        if result["success"]:
            print("Fact-checking completed successfully!")
            print(f"Result: {result['result']}")
        else:
            print(f"Fact-checking failed: {result['error']}")
    
    except Exception as e:
        print(f"Error during fact-checking: {e}")


if __name__ == "__main__":
    asyncio.run(main())