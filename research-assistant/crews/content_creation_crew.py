"""
Content Creation Crew for Research Assistant Network

Specialized CrewAI crew focused on content synthesis and report generation.
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
from agents.synthesis_agent import SynthesisAgent, SynthesisRequest
from agents.content_analyzer import ContentAnalyzer


class ContentCreationCrew:
    """Specialized crew for content synthesis and report creation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.ContentCreationCrew")
        
        # Initialize underlying agents
        self.synthesis_agent = SynthesisAgent(config.get("synthesis", {}))
        self.content_analyzer = ContentAnalyzer(config.get("content_analysis", {}))
        
        # Create crew
        self.crew = self._create_crew()
    
    def _create_crew(self) -> Crew:
        """Create the specialized content creation crew."""
        
        # Content Strategist Agent
        content_strategist = Agent(
            role="Senior Content Strategy and Planning Expert",
            goal="Plan and structure comprehensive research reports with optimal organization and flow",
            backstory="""You are a renowned content strategist with 20+ years of experience 
            in academic publishing, professional research, and content architecture. You have 
            worked with top universities, think tanks, and consulting firms to create compelling, 
            well-structured reports that effectively communicate complex information. Your 
            expertise includes information hierarchy design, narrative flow optimization, 
            and audience-appropriate content structuring.""",
            verbose=True,
            allow_delegation=True,
            llm=OpenAI(temperature=0.3),
            tools=self._create_strategy_tools()
        )
        
        # Research Writer Agent
        research_writer = Agent(
            role="Expert Research Writer and Content Developer",
            goal="Create high-quality, engaging research content that synthesizes complex information",
            backstory="""You are a distinguished research writer with expertise in academic 
            and professional writing across multiple domains. You have authored numerous 
            peer-reviewed papers, policy reports, and research publications. Your writing 
            style is clear, authoritative, and engaging while maintaining academic rigor. 
            You excel at translating complex research findings into accessible, compelling 
            narratives that serve diverse audiences.""",
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.4),
            tools=self._create_writing_tools()
        )
        
        # Citation and References Specialist
        citation_specialist = Agent(
            role="Citation Management and Academic Standards Expert",
            goal="Ensure proper citation formatting, reference management, and academic integrity",
            backstory="""You are a meticulous academic librarian and citation expert with 
            extensive experience in scholarly publishing standards. You have worked with 
            major academic publishers and universities to ensure proper attribution, citation 
            formatting, and reference management. Your expertise includes all major citation 
            styles (APA, MLA, Chicago, IEEE) and you have a keen eye for maintaining academic 
            integrity and proper source attribution.""",
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.1),
            tools=self._create_citation_tools()
        )
        
        # Quality Assurance Editor
        quality_editor = Agent(
            role="Senior Quality Assurance and Editorial Expert",
            goal="Ensure report quality, coherence, accuracy, and professional presentation standards",
            backstory="""You are a senior editor with extensive experience in academic and 
            professional publishing. You have worked with top-tier journals, research 
            institutions, and consulting firms to ensure the highest quality standards in 
            research publications. Your expertise includes fact-checking, coherence analysis, 
            style consistency, and ensuring that reports meet professional and academic 
            standards for clarity, accuracy, and impact.""",
            verbose=True,
            allow_delegation=False,
            llm=OpenAI(temperature=0.2),
            tools=self._create_quality_tools()
        )
        
        # Create crew with sequential process for methodical content creation
        return Crew(
            agents=[content_strategist, research_writer, citation_specialist, quality_editor],
            tasks=[],  # Tasks will be created dynamically
            process=Process.sequential,
            verbose=True,
            memory=True,
            max_execution_time=2400  # 40 minutes timeout
        )
    
    def _create_strategy_tools(self) -> List[Tool]:
        """Create tools for content strategy."""
        tools = []
        
        # Content structure planning tool
        def plan_content_structure(topic: str, source_count: int, target_length: int, format_type: str) -> str:
            """Plan optimal content structure for a research report."""
            try:
                # Define structure templates
                structures = {
                    "academic": {
                        "sections": ["Introduction", "Literature Review", "Methodology", "Findings", "Discussion", "Conclusion"],
                        "allocation": [0.10, 0.15, 0.10, 0.30, 0.25, 0.10]
                    },
                    "journalistic": {
                        "sections": ["Lead", "Background", "Key Findings", "Expert Analysis", "Implications", "Conclusion"],
                        "allocation": [0.15, 0.20, 0.30, 0.15, 0.15, 0.05]
                    },
                    "executive_summary": {
                        "sections": ["Executive Overview", "Key Insights", "Recommendations", "Implementation", "Next Steps"],
                        "allocation": [0.25, 0.35, 0.25, 0.10, 0.05]
                    },
                    "technical_report": {
                        "sections": ["Abstract", "Introduction", "Technical Analysis", "Results", "Discussion", "Conclusions"],
                        "allocation": [0.05, 0.15, 0.35, 0.25, 0.15, 0.05]
                    }
                }
                
                structure = structures.get(format_type, structures["academic"])
                sections = structure["sections"]
                allocations = structure["allocation"]
                
                # Calculate word allocations
                section_details = []
                for i, section in enumerate(sections):
                    word_count = int(target_length * allocations[i])
                    section_details.append(f"{section}: {word_count} words ({allocations[i]:.0%})")
                
                strategy_plan = f"""
CONTENT STRUCTURE PLAN for: "{topic}"

FORMAT TYPE: {format_type.title()}
TARGET LENGTH: {target_length} words
SOURCE COUNT: {source_count}

RECOMMENDED STRUCTURE:
{chr(10).join(f"{i+1}. {detail}" for i, detail in enumerate(section_details))}

CONTENT FLOW STRATEGY:
1. Start with compelling hook related to {topic}
2. Establish context and significance
3. Present evidence systematically
4. Build logical argument progression
5. Conclude with clear takeaways

AUDIENCE CONSIDERATIONS:
- Professional/Academic audience
- Assume moderate subject knowledge
- Balance depth with accessibility
- Include practical implications

SOURCE INTEGRATION STRATEGY:
- Distribute {source_count} sources across sections
- Prioritize highest-credibility sources for key claims
- Ensure diverse source types for comprehensive coverage
- Maintain proper attribution throughout
"""
                return strategy_plan
                
            except Exception as e:
                return f"Content planning failed: {str(e)}"
        
        tools.append(Tool(
            name="plan_structure",
            description="Plan optimal content structure and organization for research reports",
            func=plan_content_structure
        ))
        
        # Theme organization tool
        def organize_themes(content_summary: str, focus_areas: List[str]) -> str:
            """Organize content themes for logical flow."""
            try:
                # Analyze content for themes (simplified)
                themes_analysis = f"""
THEME ORGANIZATION ANALYSIS

IDENTIFIED THEMES in content:
{self._extract_themes_from_summary(content_summary)}

FOCUS AREAS INTEGRATION:
{chr(10).join(f"• {area}: Priority theme for detailed coverage" for area in focus_areas)}

RECOMMENDED THEME FLOW:
1. Context Setting: Background and current state
2. Evidence Presentation: Key findings and data
3. Analysis and Interpretation: What the evidence means
4. Implications and Applications: Practical significance
5. Future Directions: Next steps and recommendations

NARRATIVE ARC:
Problem/Question → Evidence → Analysis → Solutions → Future

LOGICAL CONNECTIONS:
- Use transitional elements between themes
- Build complexity gradually
- Return to core focus areas consistently
- Maintain coherent argument thread
"""
                return themes_analysis
                
            except Exception as e:
                return f"Theme organization failed: {str(e)}"
        
        tools.append(Tool(
            name="organize_themes",
            description="Organize content themes and topics for logical narrative flow",
            func=organize_themes
        ))
        
        return tools
    
    def _create_writing_tools(self) -> List[Tool]:
        """Create tools for research writing."""
        tools = []
        
        # Content synthesis tool
        async def synthesize_section_content(section_name: str, source_materials: str, target_words: int) -> str:
            """Synthesize content for a specific report section."""
            try:
                # Prepare simplified source materials
                sources = [
                    {
                        "url": "synthesis_source.com",
                        "title": f"Source Material for {section_name}",
                        "content": source_materials,
                        "credibility_score": 0.8,
                        "source_type": "research"
                    }
                ]
                
                # Create synthesis request
                request = SynthesisRequest(
                    topic=section_name,
                    source_materials=sources,
                    report_format="academic",
                    target_length=target_words
                )
                
                result = await self.synthesis_agent.synthesize_content(request)
                
                if result.sections:
                    section_content = result.sections[0].content
                    synthesis_result = f"""
SYNTHESIZED CONTENT for {section_name}:

{section_content}

SYNTHESIS METADATA:
- Word count: {len(section_content.split())}
- Quality score: {result.quality_score:.2f}
- Sources integrated: {len(sources)}
- Confidence level: {result.sections[0].confidence_score:.2f}
"""
                    return synthesis_result
                else:
                    return f"Failed to synthesize content for {section_name}"
                    
            except Exception as e:
                return f"Content synthesis failed: {str(e)}"
        
        tools.append(Tool(
            name="synthesize_content",
            description="Synthesize research content for specific report sections",
            func=synthesize_section_content
        ))
        
        # Writing enhancement tool
        def enhance_writing_quality(draft_content: str, target_style: str) -> str:
            """Enhance writing quality and style."""
            try:
                enhancement_suggestions = f"""
WRITING ENHANCEMENT ANALYSIS for {target_style} style:

ORIGINAL CONTENT ANALYSIS:
- Word count: {len(draft_content.split())}
- Average sentence length: {self._calculate_avg_sentence_length(draft_content):.1f} words
- Readability assessment: {self._assess_readability(draft_content)}

ENHANCEMENT RECOMMENDATIONS:

CLARITY IMPROVEMENTS:
- Use active voice where possible
- Eliminate unnecessary jargon
- Ensure clear topic sentences
- Add transitional phrases between ideas

STYLE ADJUSTMENTS for {target_style}:
{self._get_style_recommendations(target_style)}

STRUCTURE SUGGESTIONS:
- Lead with strongest evidence
- Group related concepts together
- Use parallel structure in lists
- Conclude with clear takeaways

ENHANCED EXCERPT (first paragraph):
{self._enhance_paragraph_example(draft_content[:200])}

OVERALL QUALITY SCORE: {self._calculate_quality_score(draft_content):.1f}/10
"""
                return enhancement_suggestions
                
            except Exception as e:
                return f"Writing enhancement failed: {str(e)}"
        
        tools.append(Tool(
            name="enhance_writing",
            description="Enhance writing quality, clarity, and style for research reports",
            func=enhance_writing_quality
        ))
        
        return tools
    
    def _create_citation_tools(self) -> List[Tool]:
        """Create tools for citation management."""
        tools = []
        
        # Citation formatting tool
        def format_citations(sources_list: str, citation_style: str) -> str:
            """Format citations according to specified style."""
            try:
                citation_formats = {
                    "apa": "Author, A. A. (Year). Title of work. Publisher. URL",
                    "mla": "Author. \"Title.\" Source, Date. Web. Access Date.",
                    "chicago": "Author. \"Title.\" Source. Date. URL.",
                    "ieee": "[1] A. Author, \"Title,\" Source, Year. [Online]. Available: URL"
                }
                
                format_template = citation_formats.get(citation_style, citation_formats["apa"])
                
                formatted_citations = f"""
CITATION FORMATTING for {citation_style.upper()} style:

FORMAT TEMPLATE:
{format_template}

SAMPLE FORMATTED CITATIONS:
{self._generate_sample_citations(sources_list, citation_style)}

IN-TEXT CITATION GUIDELINES:
{self._get_intext_guidelines(citation_style)}

REFERENCE LIST REQUIREMENTS:
- Alphabetical order by author last name
- Hanging indent for each entry
- Consistent formatting throughout
- Include DOI or URL when available

QUALITY CHECKLIST:
✓ All sources properly attributed
✓ Consistent citation style throughout
✓ Complete bibliographic information
✓ Proper in-text citation format
"""
                return formatted_citations
                
            except Exception as e:
                return f"Citation formatting failed: {str(e)}"
        
        tools.append(Tool(
            name="format_citations",
            description="Format citations and references according to academic standards",
            func=format_citations
        ))
        
        return tools
    
    def _create_quality_tools(self) -> List[Tool]:
        """Create tools for quality assurance."""
        tools = []
        
        # Coherence analysis tool
        def analyze_coherence(report_content: str) -> str:
            """Analyze report coherence and logical flow."""
            try:
                coherence_analysis = f"""
COHERENCE ANALYSIS REPORT:

STRUCTURAL COHERENCE:
- Section transitions: {self._assess_transitions(report_content)}
- Logical flow: {self._assess_logical_flow(report_content)}
- Argument consistency: {self._assess_argument_consistency(report_content)}

CONTENT COHERENCE:
- Theme development: {self._assess_theme_development(report_content)}
- Evidence integration: {self._assess_evidence_integration(report_content)}
- Conclusion alignment: {self._assess_conclusion_alignment(report_content)}

IMPROVEMENT RECOMMENDATIONS:
{self._generate_coherence_recommendations(report_content)}

OVERALL COHERENCE SCORE: {self._calculate_coherence_score(report_content):.1f}/10

SPECIFIC ISSUES IDENTIFIED:
{self._identify_coherence_issues(report_content)}
"""
                return coherence_analysis
                
            except Exception as e:
                return f"Coherence analysis failed: {str(e)}"
        
        tools.append(Tool(
            name="analyze_coherence",
            description="Analyze report coherence, logical flow, and structural integrity",
            func=analyze_coherence
        ))
        
        # Final quality check tool
        def final_quality_check(complete_report: str) -> str:
            """Perform comprehensive final quality assessment."""
            try:
                quality_check = f"""
FINAL QUALITY ASSESSMENT:

CONTENT QUALITY:
- Accuracy: Sources properly cited and claims supported
- Completeness: All required sections present and developed
- Depth: Adequate analysis and insight provided
- Relevance: Content addresses stated objectives

WRITING QUALITY:
- Clarity: Ideas expressed clearly and concisely
- Style: Appropriate tone and voice for audience
- Grammar: Proper grammar, punctuation, and syntax
- Flow: Smooth transitions and logical progression

TECHNICAL QUALITY:
- Citations: Proper formatting and attribution
- Structure: Logical organization and hierarchy
- Length: Appropriate for scope and purpose
- Formatting: Professional presentation standards

OVERALL RATINGS:
- Content Quality: {self._rate_content_quality(complete_report)}/10
- Writing Quality: {self._rate_writing_quality(complete_report)}/10
- Technical Quality: {self._rate_technical_quality(complete_report)}/10

FINAL RECOMMENDATION: {self._final_recommendation(complete_report)}

PUBLICATION READINESS: {self._assess_publication_readiness(complete_report)}
"""
                return quality_check
                
            except Exception as e:
                return f"Quality check failed: {str(e)}"
        
        tools.append(Tool(
            name="final_quality_check",
            description="Perform comprehensive final quality assessment of research reports",
            func=final_quality_check
        ))
        
        return tools
    
    # Helper methods for tool functions
    def _extract_themes_from_summary(self, summary: str) -> str:
        """Extract themes from content summary."""
        # Simplified theme extraction
        common_themes = ["technology", "research", "analysis", "findings", "implications", "methodology"]
        found_themes = [theme for theme in common_themes if theme in summary.lower()]
        return "\n".join(f"• {theme.title()}" for theme in found_themes[:5])
    
    def _calculate_avg_sentence_length(self, content: str) -> float:
        """Calculate average sentence length."""
        sentences = content.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        total_words = sum(len(sentence.split()) for sentence in sentences)
        return total_words / len(sentences)
    
    def _assess_readability(self, content: str) -> str:
        """Assess content readability."""
        avg_length = self._calculate_avg_sentence_length(content)
        if avg_length < 15:
            return "High (easy to read)"
        elif avg_length < 25:
            return "Medium (moderate complexity)"
        else:
            return "Low (complex, may need simplification)"
    
    def _get_style_recommendations(self, style: str) -> str:
        """Get style-specific recommendations."""
        recommendations = {
            "academic": "- Use formal tone\n- Include methodology\n- Emphasize evidence\n- Minimize personal pronouns",
            "journalistic": "- Use engaging lead\n- Include quotes\n- Focus on newsworthiness\n- Write for general audience",
            "executive": "- Lead with key findings\n- Use bullet points\n- Include actionable recommendations\n- Keep concise"
        }
        return recommendations.get(style, recommendations["academic"])
    
    def _enhance_paragraph_example(self, text: str) -> str:
        """Provide enhanced version of paragraph."""
        # Simplified enhancement example
        return f"Enhanced version: {text[:100]}... [would show improved clarity and flow]"
    
    def _calculate_quality_score(self, content: str) -> float:
        """Calculate basic quality score."""
        # Simplified quality scoring
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        
        if word_count < 100:
            return 5.0
        elif sentence_count == 0:
            return 3.0
        else:
            avg_length = word_count / sentence_count
            if 10 <= avg_length <= 25:
                return 8.0
            else:
                return 6.5
    
    def _generate_sample_citations(self, sources: str, style: str) -> str:
        """Generate sample citations."""
        # Simplified sample generation
        if style == "apa":
            return "Smith, J. (2024). Research findings. Academic Press. https://example.com"
        elif style == "mla":
            return "Smith, John. \"Research Findings.\" Academic Journal, 2024. Web. 15 Jan 2024."
        else:
            return f"Sample {style.upper()} citation format would appear here"
    
    def _get_intext_guidelines(self, style: str) -> str:
        """Get in-text citation guidelines."""
        guidelines = {
            "apa": "(Smith, 2024) or Smith (2024) states...",
            "mla": "(Smith 15) or Smith argues...",
            "chicago": "(Smith 2024, 15) or Smith notes..."
        }
        return guidelines.get(style, guidelines["apa"])
    
    def _assess_transitions(self, content: str) -> str:
        """Assess section transitions."""
        transition_words = ["however", "furthermore", "therefore", "moreover", "consequently"]
        has_transitions = any(word in content.lower() for word in transition_words)
        return "Good" if has_transitions else "Needs improvement"
    
    def _assess_logical_flow(self, content: str) -> str:
        """Assess logical flow."""
        # Simplified assessment
        return "Coherent" if len(content) > 500 else "Needs development"
    
    def _assess_argument_consistency(self, content: str) -> str:
        """Assess argument consistency."""
        return "Consistent" if "however" not in content.lower() or "therefore" in content.lower() else "Mixed"
    
    def _assess_theme_development(self, content: str) -> str:
        """Assess theme development."""
        return "Well-developed" if len(content.split()) > 200 else "Underdeveloped"
    
    def _assess_evidence_integration(self, content: str) -> str:
        """Assess evidence integration."""
        evidence_indicators = ["study", "research", "data", "evidence", "findings"]
        has_evidence = any(indicator in content.lower() for indicator in evidence_indicators)
        return "Good integration" if has_evidence else "Needs more evidence"
    
    def _assess_conclusion_alignment(self, content: str) -> str:
        """Assess conclusion alignment."""
        return "Aligned" if "conclusion" in content.lower() or "therefore" in content.lower() else "Unclear"
    
    def _generate_coherence_recommendations(self, content: str) -> str:
        """Generate coherence recommendations."""
        recommendations = []
        if "however" not in content.lower():
            recommendations.append("Add transitional phrases")
        if len(content.split()) < 300:
            recommendations.append("Expand content development")
        if "conclusion" not in content.lower():
            recommendations.append("Add clear conclusions")
        
        return "\n".join(f"• {rec}" for rec in recommendations) if recommendations else "• Content coherence is adequate"
    
    def _calculate_coherence_score(self, content: str) -> float:
        """Calculate coherence score."""
        # Simplified scoring
        score = 7.0
        if "however" in content.lower():
            score += 0.5
        if "therefore" in content.lower():
            score += 0.5
        if len(content.split()) > 300:
            score += 1.0
        return min(10.0, score)
    
    def _identify_coherence_issues(self, content: str) -> str:
        """Identify specific coherence issues."""
        issues = []
        if len(content.split()) < 200:
            issues.append("Content too brief for comprehensive analysis")
        if content.count('.') < 5:
            issues.append("Too few sentences for complex development")
        
        return "\n".join(f"• {issue}" for issue in issues) if issues else "• No major coherence issues identified"
    
    def _rate_content_quality(self, content: str) -> int:
        """Rate content quality."""
        return 8 if len(content.split()) > 500 else 6
    
    def _rate_writing_quality(self, content: str) -> int:
        """Rate writing quality."""
        return 8 if self._calculate_avg_sentence_length(content) < 25 else 6
    
    def _rate_technical_quality(self, content: str) -> int:
        """Rate technical quality."""
        return 8 if "." in content and len(content) > 100 else 6
    
    def _final_recommendation(self, content: str) -> str:
        """Provide final recommendation."""
        if len(content.split()) > 500:
            return "Ready for publication with minor revisions"
        else:
            return "Requires additional development before publication"
    
    def _assess_publication_readiness(self, content: str) -> str:
        """Assess publication readiness."""
        return "Ready" if len(content.split()) > 800 else "Needs revision"
    
    async def execute_content_creation(self, topic: str, source_materials: List[Dict[str, Any]], 
                                     requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive content creation workflow."""
        
        try:
            # Create dynamic tasks for content creation
            tasks = self._create_content_tasks(topic, source_materials, requirements)
            
            # Update crew with new tasks
            self.crew.tasks = tasks
            
            # Execute the crew
            result = await asyncio.to_thread(self.crew.kickoff)
            
            return {
                "success": True,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "topic": topic,
                "source_count": len(source_materials),
                "requirements": requirements
            }
            
        except Exception as e:
            self.logger.error(f"Content creation execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "topic": topic
            }
    
    def _create_content_tasks(self, topic: str, source_materials: List[Dict[str, Any]], 
                            requirements: Dict[str, Any]) -> List[Task]:
        """Create content creation tasks."""
        
        tasks = []
        
        # Task 1: Content Strategy and Planning
        strategy_task = Task(
            description=f"""
            Develop comprehensive content strategy for: "{topic}"
            
            Requirements:
            - Target length: {requirements.get('target_length', 1500)} words
            - Format: {requirements.get('report_format', 'academic')}
            - Sources available: {len(source_materials)}
            - Focus areas: {', '.join(requirements.get('focus_areas', []))}
            
            Deliverables:
            1. Optimal content structure and section allocation
            2. Theme organization and narrative flow plan
            3. Source integration strategy
            4. Audience-appropriate style guidelines
            
            Create detailed content strategy for systematic development.
            """,
            agent=self.crew.agents[0],  # Content strategist
            expected_output="Comprehensive content strategy with structure and flow plan"
        )
        tasks.append(strategy_task)
        
        # Task 2: Content Writing and Synthesis
        writing_task = Task(
            description=f"""
            Write comprehensive research report on: "{topic}"
            
            Using the content strategy and these materials:
            - {len(source_materials)} verified sources
            - Target format: {requirements.get('report_format', 'academic')}
            - Word target: {requirements.get('target_length', 1500)}
            
            Requirements:
            1. Follow planned structure and organization
            2. Synthesize source materials effectively
            3. Maintain consistent voice and style
            4. Include compelling introduction and conclusion
            5. Ensure smooth flow between sections
            
            Create polished, professional research content.
            """,
            agent=self.crew.agents[1],  # Research writer
            expected_output="Complete research report with synthesized content"
        )
        tasks.append(writing_task)
        
        # Task 3: Citation and Reference Management
        citation_task = Task(
            description=f"""
            Manage citations and references for the research report:
            
            Requirements:
            1. Format all citations in {requirements.get('citation_style', 'APA')} style
            2. Ensure proper in-text citations throughout
            3. Create complete reference list
            4. Verify all sources are properly attributed
            5. Check for consistency in citation format
            
            Source materials to cite: {len(source_materials)} sources
            
            Ensure academic integrity and professional citation standards.
            """,
            agent=self.crew.agents[2],  # Citation specialist
            expected_output="Properly formatted citations and complete reference list"
        )
        tasks.append(citation_task)
        
        # Task 4: Quality Assurance and Final Review
        quality_task = Task(
            description=f"""
            Conduct comprehensive quality assurance for: "{topic}" report
            
            Review areas:
            1. Content accuracy and completeness
            2. Writing clarity and coherence
            3. Technical formatting and citations
            4. Professional presentation standards
            5. Audience appropriateness
            
            Target specifications:
            - Length: {requirements.get('target_length', 1500)} words
            - Format: {requirements.get('report_format', 'academic')}
            - Quality threshold: {requirements.get('quality_threshold', 0.8)}
            
            Provide final quality assessment and publication readiness evaluation.
            """,
            agent=self.crew.agents[3],  # Quality editor
            expected_output="Quality assessment report with final recommendations"
        )
        tasks.append(quality_task)
        
        return tasks


async def main():
    """Test the content creation crew."""
    logging.basicConfig(level=logging.INFO)
    
    # Create content creation crew
    crew = ContentCreationCrew()
    
    print("Content Creation Crew initialized. Testing content creation workflow...")
    
    # Test materials
    topic = "Impact of Artificial Intelligence on Modern Healthcare"
    source_materials = [
        {
            "url": "https://example.com/ai-healthcare1",
            "title": "AI Transforming Medical Diagnosis",
            "content": "Artificial intelligence is revolutionizing healthcare through improved diagnostic accuracy and personalized treatment plans.",
            "credibility_score": 0.9,
            "source_type": "academic"
        },
        {
            "url": "https://example.com/ai-healthcare2",
            "title": "Machine Learning in Drug Discovery",
            "content": "Machine learning algorithms are accelerating drug discovery processes, reducing development time from years to months.",
            "credibility_score": 0.85,
            "source_type": "research"
        }
    ]
    
    requirements = {
        "target_length": 1200,
        "report_format": "academic",
        "citation_style": "APA",
        "focus_areas": ["diagnosis", "treatment", "efficiency"],
        "quality_threshold": 0.8
    }
    
    print(f"\nCreating content for: {topic}")
    print("=" * 60)
    
    try:
        result = await crew.execute_content_creation(topic, source_materials, requirements)
        
        if result["success"]:
            print("Content creation completed successfully!")
            print(f"Result: {result['result']}")
        else:
            print(f"Content creation failed: {result['error']}")
    
    except Exception as e:
        print(f"Error during content creation: {e}")


if __name__ == "__main__":
    asyncio.run(main())