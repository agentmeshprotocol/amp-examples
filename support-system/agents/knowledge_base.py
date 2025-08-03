"""
Knowledge Base Agent.

This agent provides instant answers from documentation, FAQs, and knowledge
articles using semantic search and retrieval-augmented generation.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import re

# AMP imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared-lib'))

from amp_client import AMPClient, AMPClientConfig
from amp_types import Capability, CapabilityConstraints, TransportType

# Support system imports
from ..support_types import Ticket, TicketCategory


class KnowledgeBaseStore:
    """Simulated knowledge base store with semantic search capabilities."""
    
    def __init__(self):
        self.articles = {
            "kb-001": {
                "id": "kb-001",
                "title": "Getting Started with User Management",
                "category": "user_management",
                "content": """User management allows you to control access and permissions for your team. 
                To add a new user: 1. Go to Settings > Users, 2. Click 'Add User', 3. Enter email and role, 
                4. Send invitation. Users can have roles: Admin, Editor, Viewer. Admins have full access, 
                Editors can modify content, Viewers have read-only access.""",
                "tags": ["users", "permissions", "roles", "access", "getting-started"],
                "last_updated": "2024-01-15",
                "rating": 4.8,
                "view_count": 1250,
                "helpful_votes": 95
            },
            "kb-002": {
                "id": "kb-002", 
                "title": "API Authentication and Rate Limits",
                "category": "api",
                "content": """API authentication requires an API key in the Authorization header. 
                Format: 'Authorization: Bearer YOUR_API_KEY'. Rate limits: Free accounts: 100 requests/hour, 
                Pro accounts: 1000 requests/hour, Enterprise: 10000 requests/hour. When rate limited, 
                you'll receive a 429 status code. Implement exponential backoff for retry logic.""",
                "tags": ["api", "authentication", "rate-limits", "bearer-token", "429-error"],
                "last_updated": "2024-01-20",
                "rating": 4.5,
                "view_count": 890,
                "helpful_votes": 67
            },
            "kb-003": {
                "id": "kb-003",
                "title": "Troubleshooting Login Issues",
                "category": "authentication",
                "content": """Common login problems and solutions: 1. Forgot password: Use 'Forgot Password' 
                link, check spam folder for reset email. 2. Account locked: Contact support after 5 failed 
                attempts. 3. Browser issues: Clear cookies and cache, try incognito mode. 4. Two-factor 
                authentication problems: Ensure device time is synchronized, try backup codes.""",
                "tags": ["login", "password", "2fa", "browser", "troubleshooting"],
                "last_updated": "2024-01-18",
                "rating": 4.9,
                "view_count": 2100,
                "helpful_votes": 187
            },
            "kb-004": {
                "id": "kb-004",
                "title": "Billing and Subscription Management",
                "category": "billing",
                "content": """Manage your subscription in Account Settings > Billing. You can: upgrade/downgrade 
                plans, update payment methods, view invoices, download receipts. Plan changes take effect 
                immediately with prorated billing. Failed payments: Update payment method within 7 days 
                to avoid service interruption. Refunds: Available within 30 days for annual plans.""",
                "tags": ["billing", "subscription", "payment", "upgrade", "refund"],
                "last_updated": "2024-01-22",
                "rating": 4.6,
                "view_count": 1450,
                "helpful_votes": 112
            },
            "kb-005": {
                "id": "kb-005",
                "title": "Data Export and Backup Best Practices",
                "category": "data_management",
                "content": """Regular data backup ensures business continuity. Export options: CSV, JSON, API. 
                Automated backups: Set up daily/weekly schedules in Settings > Data Management. Manual export: 
                Go to Data > Export, select date range and format. Large exports are processed asynchronously. 
                Retention: Free accounts: 30 days, Pro: 1 year, Enterprise: unlimited.""",
                "tags": ["data", "export", "backup", "csv", "json", "retention"],
                "last_updated": "2024-01-25",
                "rating": 4.7,
                "view_count": 780,
                "helpful_votes": 89
            }
        }
        
        self.faqs = {
            "faq-001": {
                "question": "How do I reset my password?",
                "answer": "Click 'Forgot Password' on the login page, enter your email, and check your inbox for reset instructions.",
                "category": "authentication",
                "popularity": 150
            },
            "faq-002": {
                "question": "What are the API rate limits?",
                "answer": "Free: 100/hour, Pro: 1000/hour, Enterprise: 10000/hour. You'll get a 429 status when limits are exceeded.",
                "category": "api",
                "popularity": 95
            },
            "faq-003": {
                "question": "How do I upgrade my subscription?",
                "answer": "Go to Account Settings > Billing > Change Plan. Select your new plan and confirm. Changes are immediate with prorated billing.",
                "category": "billing",
                "popularity": 120
            },
            "faq-004": {
                "question": "Can I export my data?",
                "answer": "Yes, go to Data > Export and choose your format (CSV, JSON). Large exports are processed in the background.",
                "category": "data_management",
                "popularity": 80
            }
        }
        
        # Search index for semantic matching
        self.search_index = self._build_search_index()
    
    def _build_search_index(self) -> Dict[str, List[str]]:
        """Build a simple search index."""
        index = {}
        
        # Index articles
        for article_id, article in self.articles.items():
            terms = self._extract_search_terms(article["content"] + " " + " ".join(article["tags"]))
            index[article_id] = terms
        
        # Index FAQs
        for faq_id, faq in self.faqs.items():
            terms = self._extract_search_terms(faq["question"] + " " + faq["answer"])
            index[faq_id] = terms
        
        return index
    
    def _extract_search_terms(self, text: str) -> List[str]:
        """Extract search terms from text."""
        # Simple term extraction - in production, use proper NLP
        text = text.lower()
        terms = re.findall(r'\b\w+\b', text)
        return list(set(terms))
    
    def search_articles(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search knowledge base articles."""
        query_terms = self._extract_search_terms(query)
        results = []
        
        for article_id, article in self.articles.items():
            if category and article["category"] != category:
                continue
            
            # Calculate relevance score
            article_terms = self.search_index.get(article_id, [])
            common_terms = set(query_terms) & set(article_terms)
            relevance = len(common_terms) / max(len(query_terms), 1)
            
            if relevance > 0.1:  # Minimum relevance threshold
                results.append({
                    **article,
                    "relevance_score": relevance,
                    "matched_terms": list(common_terms)
                })
        
        # Sort by relevance and popularity
        results.sort(key=lambda x: (x["relevance_score"], x["rating"]), reverse=True)
        return results[:10]
    
    def search_faqs(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search FAQ entries."""
        query_terms = self._extract_search_terms(query)
        results = []
        
        for faq_id, faq in self.faqs.items():
            if category and faq["category"] != category:
                continue
            
            # Calculate relevance score
            faq_terms = self.search_index.get(faq_id, [])
            common_terms = set(query_terms) & set(faq_terms)
            relevance = len(common_terms) / max(len(query_terms), 1)
            
            if relevance > 0.1:
                results.append({
                    **faq,
                    "id": faq_id,
                    "relevance_score": relevance,
                    "matched_terms": list(common_terms)
                })
        
        # Sort by relevance and popularity
        results.sort(key=lambda x: (x["relevance_score"], x["popularity"]), reverse=True)
        return results[:5]
    
    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get specific article by ID."""
        return self.articles.get(article_id)
    
    def get_related_articles(self, article_id: str) -> List[Dict[str, Any]]:
        """Get articles related to the given article."""
        article = self.articles.get(article_id)
        if not article:
            return []
        
        # Find articles with similar tags
        article_tags = set(article["tags"])
        related = []
        
        for aid, art in self.articles.items():
            if aid == article_id:
                continue
            
            common_tags = article_tags & set(art["tags"])
            if len(common_tags) >= 2:  # At least 2 common tags
                related.append({
                    **art,
                    "common_tags": list(common_tags)
                })
        
        return related[:3]
    
    def update_article_metrics(self, article_id: str, action: str):
        """Update article metrics (views, votes, etc.)."""
        if article_id in self.articles:
            if action == "view":
                self.articles[article_id]["view_count"] += 1
            elif action == "helpful":
                self.articles[article_id]["helpful_votes"] += 1


class KnowledgeBaseAgent:
    """
    Knowledge Base Agent for providing instant answers from documentation
    and knowledge articles.
    """
    
    def __init__(self, config: AMPClientConfig):
        self.config = config
        self.client = AMPClient(config)
        self.logger = logging.getLogger(f"support.knowledge.{config.agent_id}")
        
        # Initialize knowledge base
        self.kb_store = KnowledgeBaseStore()
        
        # Category mapping for better search
        self.category_mapping = {
            TicketCategory.TECHNICAL: ["authentication", "api", "data_management"],
            TicketCategory.BILLING: ["billing"],
            TicketCategory.PRODUCT: ["user_management", "data_management"],
            TicketCategory.ACCOUNT: ["user_management", "authentication"],
        }
        
    async def start(self):
        """Start the knowledge base agent."""
        await self._register_capabilities()
        
        connected = await self.client.connect()
        if not connected:
            raise RuntimeError("Failed to connect to AMP network")
        
        self.logger.info("Knowledge Base Agent started successfully")
    
    async def stop(self):
        """Stop the knowledge base agent."""
        await self.client.disconnect()
        self.logger.info("Knowledge Base Agent stopped")
    
    async def _register_capabilities(self):
        """Register agent capabilities with AMP."""
        
        # Knowledge search capability
        knowledge_search_capability = Capability(
            id="knowledge-search",
            version="1.0",
            description="Search knowledge base for instant answers to common questions",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string"},
                    "search_type": {"type": "string", "enum": ["all", "articles", "faqs"]},
                    "max_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "search_results": {"type": "array"},
                    "suggested_articles": {"type": "array"},
                    "quick_answers": {"type": "array"},
                    "search_metadata": {"type": "object"}
                }
            },
            constraints=CapabilityConstraints(
                response_time_ms=3000,
                max_input_length=1000,
                supported_languages=["en"],
                min_confidence=0.6
            ),
            category="knowledge-retrieval",
            subcategories=["search", "documentation", "faq"]
        )
        
        self.client.register_capability(knowledge_search_capability, self.search_knowledge)
        
        # Answer generation capability
        answer_generation_capability = Capability(
            id="answer-generation",
            version="1.0",
            description="Generate comprehensive answers using knowledge base content",
            input_schema={
                "type": "object",
                "properties": {
                    "ticket": {"type": "object"},
                    "context": {"type": "object"},
                    "include_related": {"type": "boolean", "default": True}
                },
                "required": ["ticket"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "generated_answer": {"type": "object"},
                    "source_articles": {"type": "array"},
                    "confidence_score": {"type": "number"},
                    "follow_up_suggestions": {"type": "array"}
                }
            },
            constraints=CapabilityConstraints(
                response_time_ms=5000,
                max_input_length=5000
            )
        )
        
        self.client.register_capability(answer_generation_capability, self.generate_answer)
    
    async def search_knowledge(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search the knowledge base for relevant information.
        
        Args:
            parameters: Contains search query and filters
            
        Returns:
            Search results with articles, FAQs, and metadata
        """
        try:
            query = parameters["query"]
            category = parameters.get("category")
            search_type = parameters.get("search_type", "all")
            max_results = parameters.get("max_results", 5)
            
            search_results = []
            
            # Search articles
            if search_type in ["all", "articles"]:
                articles = self.kb_store.search_articles(query, category)
                for article in articles[:max_results]:
                    search_results.append({
                        "type": "article",
                        "id": article["id"],
                        "title": article["title"],
                        "content_snippet": article["content"][:200] + "...",
                        "relevance_score": article["relevance_score"],
                        "category": article["category"],
                        "url": f"/kb/articles/{article['id']}",
                        "rating": article["rating"],
                        "view_count": article["view_count"]
                    })
            
            # Search FAQs
            if search_type in ["all", "faqs"]:
                faqs = self.kb_store.search_faqs(query, category)
                for faq in faqs:
                    search_results.append({
                        "type": "faq",
                        "id": faq["id"],
                        "question": faq["question"],
                        "answer": faq["answer"],
                        "relevance_score": faq["relevance_score"],
                        "category": faq["category"],
                        "popularity": faq["popularity"]
                    })
            
            # Sort all results by relevance
            search_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            search_results = search_results[:max_results]
            
            # Generate quick answers from top results
            quick_answers = self._generate_quick_answers(search_results[:3])
            
            # Suggest related articles
            suggested_articles = self._suggest_related_articles(search_results, query)
            
            # Search metadata
            search_metadata = {
                "query": query,
                "total_results": len(search_results),
                "search_time_ms": 50,  # Simulated
                "categories_searched": [category] if category else ["all"],
                "search_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            response = {
                "search_results": search_results,
                "quick_answers": quick_answers,
                "suggested_articles": suggested_articles,
                "search_metadata": search_metadata,
                "agent_id": self.config.agent_id
            }
            
            self.logger.info(f"Knowledge search for '{query}' returned {len(search_results)} results")
            return response
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {e}")
            raise
    
    async def generate_answer(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive answer for a ticket using knowledge base.
        
        Args:
            parameters: Contains ticket information and context
            
        Returns:
            Generated answer with sources and confidence
        """
        try:
            ticket_data = parameters["ticket"]
            context = parameters.get("context", {})
            include_related = parameters.get("include_related", True)
            
            # Extract key information from ticket
            query = f"{ticket_data['subject']} {ticket_data['description']}"
            category = self._map_ticket_category(ticket_data.get("category"))
            
            # Search for relevant knowledge
            search_params = {
                "query": query,
                "category": category,
                "search_type": "all",
                "max_results": 10
            }
            
            search_results = await self.search_knowledge(search_params)
            
            # Generate comprehensive answer
            generated_answer = self._create_comprehensive_answer(
                ticket_data, search_results["search_results"], context
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_answer_confidence(
                search_results["search_results"], generated_answer
            )
            
            # Get source articles
            source_articles = [
                result for result in search_results["search_results"]
                if result["type"] == "article"
            ][:5]
            
            # Generate follow-up suggestions
            follow_up_suggestions = self._generate_follow_up_suggestions(
                ticket_data, search_results["search_results"]
            )
            
            # Update article metrics for viewed articles
            for article in source_articles:
                self.kb_store.update_article_metrics(article["id"], "view")
            
            response = {
                "ticket_id": ticket_data.get("id"),
                "agent_id": self.config.agent_id,
                "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                "generated_answer": generated_answer,
                "source_articles": source_articles,
                "confidence_score": confidence_score,
                "follow_up_suggestions": follow_up_suggestions,
                "search_query_used": query,
                "knowledge_coverage": self._assess_knowledge_coverage(query, search_results["search_results"])
            }
            
            self.logger.info(f"Generated answer for ticket {ticket_data.get('id')} "
                           f"(confidence: {confidence_score:.2f})")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            raise
    
    def _map_ticket_category(self, ticket_category: str) -> Optional[str]:
        """Map ticket category to knowledge base categories."""
        if not ticket_category:
            return None
        
        try:
            category_enum = TicketCategory(ticket_category)
            categories = self.category_mapping.get(category_enum, [])
            return categories[0] if categories else None
        except ValueError:
            return None
    
    def _generate_quick_answers(self, top_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate quick answers from top search results."""
        quick_answers = []
        
        for result in top_results:
            if result["type"] == "faq":
                quick_answers.append({
                    "question": result["question"],
                    "answer": result["answer"],
                    "source": "FAQ",
                    "confidence": result["relevance_score"]
                })
            elif result["type"] == "article" and result["relevance_score"] > 0.7:
                # Extract key information from article
                content = result.get("content_snippet", "")
                if len(content) > 100:
                    quick_answers.append({
                        "question": f"Information about {result['title']}",
                        "answer": content,
                        "source": "Knowledge Base Article",
                        "confidence": result["relevance_score"]
                    })
        
        return quick_answers[:3]
    
    def _suggest_related_articles(self, search_results: List[Dict[str, Any]], 
                                query: str) -> List[Dict[str, str]]:
        """Suggest related articles based on search results."""
        suggestions = []
        
        # Get related articles for top articles
        for result in search_results[:2]:
            if result["type"] == "article":
                related = self.kb_store.get_related_articles(result["id"])
                for article in related:
                    suggestions.append({
                        "id": article["id"],
                        "title": article["title"],
                        "category": article["category"],
                        "url": f"/kb/articles/{article['id']}",
                        "reason": "Related to search results"
                    })
        
        # Remove duplicates and limit
        seen_ids = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion["id"] not in seen_ids:
                unique_suggestions.append(suggestion)
                seen_ids.add(suggestion["id"])
        
        return unique_suggestions[:5]
    
    def _create_comprehensive_answer(self, ticket_data: Dict[str, Any], 
                                   search_results: List[Dict[str, Any]],
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive answer using search results."""
        
        # Categorize search results
        articles = [r for r in search_results if r["type"] == "article"]
        faqs = [r for r in search_results if r["type"] == "faq"]
        
        # Build comprehensive answer
        answer = {
            "summary": self._generate_answer_summary(ticket_data, articles, faqs),
            "step_by_step_solution": self._extract_solution_steps(articles),
            "quick_solutions": [faq["answer"] for faq in faqs[:2]],
            "detailed_information": self._compile_detailed_info(articles[:3]),
            "additional_resources": self._compile_resources(articles),
            "troubleshooting_tips": self._extract_troubleshooting_tips(articles),
            "best_practices": self._extract_best_practices(articles)
        }
        
        return answer
    
    def _generate_answer_summary(self, ticket_data: Dict[str, Any], 
                               articles: List[Dict[str, Any]], 
                               faqs: List[Dict[str, Any]]) -> str:
        """Generate answer summary."""
        subject = ticket_data.get("subject", "your question")
        
        if faqs:
            return f"Based on our knowledge base, here's what we found regarding {subject}: " + faqs[0]["answer"]
        elif articles:
            return f"I found relevant information about {subject} in our documentation. " + articles[0].get("content_snippet", "")
        else:
            return f"I understand you're asking about {subject}. Let me help you find the right information."
    
    def _extract_solution_steps(self, articles: List[Dict[str, Any]]) -> List[str]:
        """Extract solution steps from articles."""
        steps = []
        
        for article in articles[:2]:
            content = article.get("content_snippet", "") + article.get("content", "")
            
            # Look for numbered steps
            step_patterns = [
                r'\d+\.\s+([^.]+\.)',
                r'Step \d+:\s+([^.]+\.)',
                r'First,?\s+([^.]+\.)',
                r'Next,?\s+([^.]+\.)',
                r'Then,?\s+([^.]+\.)',
                r'Finally,?\s+([^.]+\.)'
            ]
            
            for pattern in step_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                steps.extend(matches[:3])  # Limit to 3 steps per pattern
                
                if len(steps) >= 5:  # Limit total steps
                    break
            
            if len(steps) >= 5:
                break
        
        return steps[:5]
    
    def _compile_detailed_info(self, articles: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Compile detailed information from articles."""
        detailed_info = []
        
        for article in articles:
            detailed_info.append({
                "title": article["title"],
                "category": article["category"],
                "summary": article.get("content_snippet", ""),
                "full_article_url": article.get("url", ""),
                "rating": str(article.get("rating", "N/A"))
            })
        
        return detailed_info
    
    def _compile_resources(self, articles: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Compile additional resources."""
        resources = []
        
        for article in articles:
            resources.append({
                "title": article["title"],
                "type": "Knowledge Base Article",
                "url": article.get("url", ""),
                "description": f"Learn more about {article['category']}"
            })
        
        # Add some standard resources
        resources.extend([
            {
                "title": "Contact Support",
                "type": "Support Channel",
                "url": "/support/contact",
                "description": "Get personalized help from our support team"
            },
            {
                "title": "Community Forum",
                "type": "Community",
                "url": "/community",
                "description": "Connect with other users and share experiences"
            }
        ])
        
        return resources[:5]
    
    def _extract_troubleshooting_tips(self, articles: List[Dict[str, Any]]) -> List[str]:
        """Extract troubleshooting tips from articles."""
        tips = []
        
        tip_keywords = ["tip", "suggestion", "recommendation", "best practice", "avoid"]
        
        for article in articles:
            content = article.get("content_snippet", "") + article.get("content", "")
            sentences = content.split('. ')
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in tip_keywords):
                    tips.append(sentence.strip())
                    if len(tips) >= 3:
                        break
            
            if len(tips) >= 3:
                break
        
        return tips
    
    def _extract_best_practices(self, articles: List[Dict[str, Any]]) -> List[str]:
        """Extract best practices from articles."""
        practices = []
        
        practice_patterns = [
            r'Best practice[^.]*\.',
            r'Recommended[^.]*\.',
            r'Should[^.]*\.',
            r'Always[^.]*\.',
            r'Never[^.]*\.'
        ]
        
        for article in articles:
            content = article.get("content_snippet", "") + article.get("content", "")
            
            for pattern in practice_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                practices.extend(matches[:2])
                
                if len(practices) >= 3:
                    break
            
            if len(practices) >= 3:
                break
        
        return practices
    
    def _calculate_answer_confidence(self, search_results: List[Dict[str, Any]], 
                                   generated_answer: Dict[str, Any]) -> float:
        """Calculate confidence score for the generated answer."""
        if not search_results:
            return 0.1
        
        # Base confidence on search result relevance
        avg_relevance = sum(r["relevance_score"] for r in search_results[:3]) / min(len(search_results), 3)
        
        # Adjust based on answer completeness
        answer_completeness = 0.0
        if generated_answer.get("step_by_step_solution"):
            answer_completeness += 0.3
        if generated_answer.get("quick_solutions"):
            answer_completeness += 0.2
        if generated_answer.get("detailed_information"):
            answer_completeness += 0.3
        if generated_answer.get("troubleshooting_tips"):
            answer_completeness += 0.2
        
        # Combine scores
        confidence = (avg_relevance * 0.7) + (answer_completeness * 0.3)
        return min(confidence, 0.95)  # Cap at 95%
    
    def _generate_follow_up_suggestions(self, ticket_data: Dict[str, Any], 
                                      search_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate follow-up suggestions."""
        suggestions = []
        
        # Based on ticket category
        category = ticket_data.get("category")
        if category == "technical":
            suggestions.extend([
                {
                    "action": "try_troubleshooting_steps",
                    "description": "Try the troubleshooting steps provided",
                    "priority": "high"
                },
                {
                    "action": "check_system_status",
                    "description": "Check our system status page for known issues",
                    "priority": "medium"
                }
            ])
        elif category == "billing":
            suggestions.extend([
                {
                    "action": "review_billing_settings", 
                    "description": "Review your billing settings and payment methods",
                    "priority": "high"
                },
                {
                    "action": "contact_billing_support",
                    "description": "Contact our billing support for account-specific help",
                    "priority": "medium"
                }
            ])
        
        # Generic suggestions
        suggestions.extend([
            {
                "action": "read_related_articles",
                "description": "Read the related articles for more detailed information",
                "priority": "medium"
            },
            {
                "action": "contact_support",
                "description": "Contact support if the issue persists",
                "priority": "low"
            }
        ])
        
        return suggestions[:4]
    
    def _assess_knowledge_coverage(self, query: str, 
                                 search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess how well the knowledge base covers the query."""
        
        if not search_results:
            coverage = "poor"
            gap_analysis = ["No relevant articles found", "Consider creating new documentation"]
        elif search_results[0]["relevance_score"] > 0.8:
            coverage = "excellent"
            gap_analysis = ["Query well covered by existing documentation"]
        elif search_results[0]["relevance_score"] > 0.6:
            coverage = "good"
            gap_analysis = ["Partial coverage available", "Could benefit from more specific documentation"]
        else:
            coverage = "fair"
            gap_analysis = ["Limited coverage", "Consider expanding documentation for this topic"]
        
        return {
            "coverage_level": coverage,
            "gap_analysis": gap_analysis,
            "top_result_relevance": search_results[0]["relevance_score"] if search_results else 0,
            "total_relevant_articles": len([r for r in search_results if r["relevance_score"] > 0.5])
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        config = AMPClientConfig(
            agent_id="knowledge-base-001",
            agent_name="Knowledge Base Agent",
            framework="custom",
            transport_type=TransportType.HTTP,
            endpoint="http://localhost:8000"
        )
        
        agent = KnowledgeBaseAgent(config)
        
        try:
            await agent.start()
            print("Knowledge Base Agent is running...")
            
            # Test search
            test_search = await agent.search_knowledge({
                "query": "how to reset password",
                "max_results": 3
            })
            print(f"Test search returned {len(test_search['search_results'])} results")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            await agent.stop()
    
    asyncio.run(main())