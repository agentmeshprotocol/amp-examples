#!/usr/bin/env python3
"""
Research Assistant CLI Interface

Interactive command-line interface for the Research Assistant Network.
Provides easy access to research capabilities and workflows.
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import research system
from run_research_assistant import ResearchAssistantSystem


class ResearchCLI:
    """Interactive CLI for research assistant."""
    
    def __init__(self):
        self.system: Optional[ResearchAssistantSystem] = None
        self.session_history = []
    
    def print_banner(self):
        """Print CLI banner."""
        print("\\n" + "=" * 70)
        print("🤖 Research Assistant Network - Interactive CLI")
        print("=" * 70)
        print("Welcome to the AI-powered research assistant!")
        print("Type 'help' for available commands or 'quit' to exit.")
        print("=" * 70 + "\\n")
    
    def print_help(self):
        """Print help information."""
        help_text = """
Available Commands:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 RESEARCH COMMANDS:
  research <query>          - Conduct comprehensive research
  quick <query>             - Quick research (basic analysis)
  fact-check <claim>        - Verify a factual claim
  summarize <url>           - Summarize content from URL

📊 ANALYSIS COMMANDS:
  analyze <text>            - Analyze text content
  extract <url>             - Extract and analyze web content
  keywords <text>           - Extract keywords from text

⚙️  SYSTEM COMMANDS:
  status                    - Show system status
  health                    - System health check
  metrics                   - Performance metrics
  config                    - Show configuration

🛠️  UTILITY COMMANDS:
  history                   - Show session history
  clear                     - Clear screen
  help                      - Show this help
  quit/exit                 - Exit the CLI

📝 EXAMPLES:
  research "climate change impact on agriculture"
  quick "latest AI developments"
  fact-check "vaccines are 95% effective"
  summarize "https://example.com/article"

💡 TIP: Use quotes for multi-word queries and claims
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        print(help_text)
    
    async def initialize_system(self, config_path: str = "config"):
        """Initialize the research assistant system."""
        print("🚀 Initializing Research Assistant Network...")
        
        try:
            self.system = ResearchAssistantSystem(config_path=config_path)
            await self.system.start()
            print("✅ System initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            return False
    
    async def handle_research(self, query: str, quick: bool = False) -> None:
        """Handle research command."""
        if not self.system:
            print("❌ System not initialized")
            return
        
        print(f"\\n🔍 {'Quick research' if quick else 'Comprehensive research'}: {query}")
        print("━" * 60)
        print("⏳ Searching and analyzing sources...")
        
        start_time = datetime.now()
        
        try:
            parameters = {
                "depth": "basic" if quick else "standard",
                "max_sources": 5 if quick else 10,
                "focus_areas": [],
                "report_format": "executive_summary" if quick else "academic",
                "target_length": 800 if quick else 1500,
                "include_fact_checking": not quick,
                "quality_threshold": 0.5 if quick else 0.6
            }
            
            result = await self.system.conduct_research(query, parameters)
            
            if result["success"]:
                research_result = result["result"]
                duration = (datetime.now() - start_time).total_seconds()
                
                print(f"\\n✅ Research completed in {duration:.1f} seconds")
                print("━" * 60)
                
                # Display results
                print(f"📊 **{research_result.title}**")
                print(f"\\n📝 **Executive Summary:**")
                print(research_result.executive_summary)
                
                if research_result.conclusions:
                    print(f"\\n🎯 **Key Conclusions:**")
                    for i, conclusion in enumerate(research_result.conclusions[:3], 1):
                        print(f"{i}. {conclusion}")
                
                if research_result.recommendations:
                    print(f"\\n💡 **Recommendations:**")
                    for i, rec in enumerate(research_result.recommendations[:3], 1):
                        print(f"{i}. {rec}")
                
                print(f"\\n📈 **Research Metrics:**")
                print(f"• Quality Score: {research_result.quality_score:.2f}/1.0")
                print(f"• Word Count: {research_result.word_count:,}")
                print(f"• Sources Analyzed: {len(research_result.sources)}")
                print(f"• Fact Checks: {len(research_result.fact_check_results)}")
                
                # Add to history
                self.session_history.append({
                    "type": "research",
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "duration": duration,
                    "success": True
                })
                
            else:
                print(f"\\n❌ Research failed: {result['error']}")
                self.session_history.append({
                    "type": "research",
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "success": False,
                    "error": result['error']
                })
                
        except Exception as e:
            print(f"\\n❌ Error during research: {e}")
    
    async def handle_fact_check(self, claim: str) -> None:
        """Handle fact-checking command."""
        if not self.system or not self.system.agents.get("fact_checker"):
            print("❌ Fact checker not available")
            return
        
        print(f"\\n🔎 Fact-checking claim: {claim}")
        print("━" * 60)
        print("⏳ Verifying claim against multiple sources...")
        
        try:
            from agents.fact_checker import Claim
            
            fact_checker = self.system.agents["fact_checker"]
            claim_obj = Claim(text=claim, claim_type="general")
            
            result = await fact_checker.verify_claim(claim_obj)
            
            print(f"\\n📋 **Fact Check Result**")
            print(f"**Claim:** {claim}")
            print(f"**Verdict:** {result.verdict.upper()}")
            print(f"**Confidence:** {result.confidence_score:.2f}/1.0")
            
            if result.supporting_sources:
                print(f"\\n✅ **Supporting Evidence ({len(result.supporting_sources)} sources):**")
                for src in result.supporting_sources[:3]:
                    print(f"• {src.get('title', 'Unknown')[:80]}...")
                    print(f"  Credibility: {src.get('credibility_score', 0):.2f}")
            
            if result.contradicting_sources:
                print(f"\\n❌ **Contradicting Evidence ({len(result.contradicting_sources)} sources):**")
                for src in result.contradicting_sources[:3]:
                    print(f"• {src.get('title', 'Unknown')[:80]}...")
                    print(f"  Credibility: {src.get('credibility_score', 0):.2f}")
            
            print(f"\\n📊 **Verification Details:**")
            print(f"• Sources Checked: {result.verification_details.get('sources_checked', 0)}")
            print(f"• Search Queries: {len(result.verification_details.get('search_queries', []))}")
            
        except Exception as e:
            print(f"\\n❌ Error during fact-checking: {e}")
    
    async def handle_analyze(self, text: str) -> None:
        """Handle content analysis command."""
        if not self.system or not self.system.agents.get("content_analyzer"):
            print("❌ Content analyzer not available")
            return
        
        print(f"\\n📊 Analyzing content ({len(text)} characters)")
        print("━" * 60)
        
        try:
            from agents.content_analyzer import AnalysisRequest
            
            analyzer = self.system.agents["content_analyzer"]
            request = AnalysisRequest(
                content=text,
                analysis_depth="standard"
            )
            
            analysis = await analyzer.analyze_content(request)
            
            print(f"\\n📝 **Content Analysis Results**")
            print(f"**Word Count:** {analysis.word_count}")
            print(f"**Readability Score:** {analysis.readability_score:.1f}")
            print(f"**Sentiment:** {analysis.sentiment}")
            
            if analysis.key_points:
                print(f"\\n🎯 **Key Points:**")
                for i, point in enumerate(analysis.key_points[:5], 1):
                    print(f"{i}. {point}")
            
            if analysis.keywords:
                print(f"\\n🏷️  **Top Keywords:**")
                for word, score in analysis.keywords[:10]:
                    print(f"• {word}: {score:.3f}")
            
            if analysis.entities:
                print(f"\\n🏢 **Named Entities:**")
                entity_counts = {}
                for entity in analysis.entities:
                    entity_type = entity['label']
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                
                for entity_type, count in sorted(entity_counts.items()):
                    print(f"• {entity_type}: {count}")
            
        except Exception as e:
            print(f"\\n❌ Error during analysis: {e}")
    
    async def handle_status(self) -> None:
        """Handle status command."""
        if not self.system:
            print("❌ System not initialized")
            return
        
        try:
            health = await self.system.health_check()
            
            print(f"\\n🚦 **System Status: {health['status'].upper()}**")
            print("━" * 60)
            print(f"⏰ Uptime: {health['uptime_seconds']:.0f} seconds")
            print(f"📊 Components: {len(health['components'])}")
            
            print(f"\\n🔧 **Component Status:**")
            for component, info in health['components'].items():
                status_icon = "✅" if info['status'] == 'healthy' else "⚠️"
                print(f"{status_icon} {component}: {info['status']}")
            
            if health['metrics']:
                print(f"\\n📈 **Performance Metrics:**")
                metrics = health['metrics']
                print(f"• Requests Processed: {metrics['requests_processed']}")
                print(f"• Success Rate: {metrics['successful_requests']}/{metrics['requests_processed']}")
                print(f"• Average Response Time: {metrics['average_response_time']:.2f}s")
            
        except Exception as e:
            print(f"\\n❌ Error getting status: {e}")
    
    def handle_history(self) -> None:
        """Handle history command."""
        if not self.session_history:
            print("\\n📝 No session history available")
            return
        
        print(f"\\n📜 **Session History ({len(self.session_history)} items)**")
        print("━" * 60)
        
        for i, item in enumerate(reversed(self.session_history[-10:]), 1):
            timestamp = datetime.fromisoformat(item['timestamp']).strftime('%H:%M:%S')
            status = "✅" if item['success'] else "❌"
            print(f"{i}. [{timestamp}] {status} {item['type']}: {item.get('query', 'N/A')[:50]}...")
    
    async def run_interactive(self) -> None:
        """Run interactive CLI mode."""
        self.print_banner()
        
        # Initialize system
        if not await self.initialize_system():
            return
        
        try:
            while True:
                try:
                    # Get user input
                    user_input = input("\\n🤖 research> ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Parse command
                    parts = user_input.split(' ', 1)
                    command = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""
                    
                    # Handle commands
                    if command in ['quit', 'exit']:
                        print("\\n👋 Goodbye!")
                        break
                    elif command == 'help':
                        self.print_help()
                    elif command == 'clear':
                        print("\\033[2J\\033[H")  # Clear screen
                    elif command == 'research':
                        if args:
                            await self.handle_research(args)
                        else:
                            print("❌ Please provide a research query")
                    elif command == 'quick':
                        if args:
                            await self.handle_research(args, quick=True)
                        else:
                            print("❌ Please provide a research query")
                    elif command == 'fact-check':
                        if args:
                            await self.handle_fact_check(args)
                        else:
                            print("❌ Please provide a claim to fact-check")
                    elif command == 'analyze':
                        if args:
                            await self.handle_analyze(args)
                        else:
                            print("❌ Please provide text to analyze")
                    elif command == 'status':
                        await self.handle_status()
                    elif command == 'history':
                        self.handle_history()
                    else:
                        print(f"❌ Unknown command: {command}. Type 'help' for available commands.")
                
                except KeyboardInterrupt:
                    print("\\n\\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\\n\\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"\\n❌ Error: {e}")
        
        finally:
            if self.system:
                await self.system.shutdown()


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Research Assistant CLI")
    parser.add_argument("--config", help="Configuration directory", default="config")
    parser.add_argument("--query", help="Single query mode")
    parser.add_argument("--fact-check", help="Single fact-check mode")
    parser.add_argument("--quick", help="Quick research mode", action="store_true")
    
    args = parser.parse_args()
    
    cli = ResearchCLI()
    
    if args.query:
        # Single query mode
        print("🤖 Research Assistant - Single Query Mode")
        if await cli.initialize_system(args.config):
            await cli.handle_research(args.query, quick=args.quick)
            await cli.system.shutdown()
    elif args.fact_check:
        # Single fact-check mode
        print("🤖 Research Assistant - Fact Check Mode")
        if await cli.initialize_system(args.config):
            await cli.handle_fact_check(args.fact_check)
            await cli.system.shutdown()
    else:
        # Interactive mode
        await cli.run_interactive()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n👋 Goodbye!")
    except Exception as e:
        print(f"CLI Error: {e}")
        sys.exit(1)