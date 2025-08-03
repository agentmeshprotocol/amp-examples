#!/usr/bin/env python3
"""
Sample Data Generator for AMP Support System

Creates realistic sample tickets, customers, and knowledge base content
for testing and demonstration purposes.
"""

import json
import random
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import os
import sys

# Add support system to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from support_types import (
    Ticket, TicketStatus, TicketPriority, TicketCategory,
    CustomerInfo, SLALevel
)


class SampleDataGenerator:
    """Generate realistic sample data for the support system."""
    
    def __init__(self):
        self.customer_names = [
            "John Smith", "Sarah Johnson", "Michael Brown", "Emma Davis", "James Wilson",
            "Lisa Anderson", "David Miller", "Jennifer Taylor", "Robert Thomas", "Maria Garcia",
            "William Martinez", "Elizabeth Robinson", "Christopher Clark", "Patricia Rodriguez",
            "Daniel Lewis", "Barbara Walker", "Matthew Hall", "Susan Allen", "Anthony Young",
            "Margaret King", "Mark Wright", "Dorothy Lopez", "Steven Hill", "Helen Scott"
        ]
        
        self.company_domains = [
            "techcorp.com", "innovate-solutions.net", "digital-dynamics.io", "future-systems.org",
            "alpha-enterprises.biz", "beta-technologies.com", "gamma-solutions.net", "delta-corp.co",
            "epsilon-systems.org", "zeta-digital.com", "eta-innovations.net", "theta-tech.io"
        ]
        
        self.technical_subjects = [
            "Login authentication failing repeatedly",
            "API integration returning 500 errors",
            "Dashboard not loading data correctly",
            "File upload functionality broken",
            "Payment processing timeout issues",
            "SSL certificate validation errors",
            "Mobile app crashes on startup",
            "Database connection timeouts",
            "Email notifications not sending",
            "Two-factor authentication not working",
            "Performance degradation in reporting module",
            "Webhook endpoints returning 404 errors",
            "Search functionality returning no results",
            "Export feature generating corrupted files",
            "User permissions not saving correctly"
        ]
        
        self.billing_subjects = [
            "Incorrect billing amount charged",
            "Need to update payment method",
            "Request refund for duplicate charge",
            "Unable to download invoice",
            "Subscription not upgraded after payment",
            "Billing cycle date change request",
            "Credit card payment failed notification",
            "Annual subscription discount not applied",
            "Need receipt for accounting purposes",
            "Billing address update required",
            "Proration calculation seems incorrect",
            "Auto-renewal settings not working",
            "Payment method declined unexpectedly",
            "Need to cancel subscription immediately",
            "Billing history shows incorrect charges"
        ]
        
        self.product_subjects = [
            "How to set up user permissions",
            "Need training on advanced features",
            "Request for mobile app enhancement",
            "Documentation unclear for API integration",
            "Feature request: bulk data import",
            "How to configure automated workflows",
            "Need guidance on best practices",
            "Request tutorial for new team members",
            "How to customize dashboard layouts",
            "Integration with third-party tools",
            "Advanced reporting capabilities needed",
            "How to backup and restore data",
            "Multi-language support request",
            "Custom field configuration help",
            "Workflow automation setup guidance"
        ]
        
        self.technical_descriptions = [
            "I've been trying to log into my account for the past 2 hours but keep getting an 'Authentication Failed' error. I've tried resetting my password multiple times but the issue persists. This is blocking my work and I need urgent assistance.",
            
            "Our API integration suddenly started failing yesterday with 500 Internal Server Error responses. The error occurs on all endpoints we're calling. Our production system is affected and we need immediate support to resolve this issue.",
            
            "The main dashboard is loading but not displaying any data. All widgets show loading spinners indefinitely. This started happening after the last system update. Multiple users are reporting the same issue.",
            
            "File upload feature stopped working completely. When users try to upload documents, they get a 'Upload Failed' message with no additional details. This is affecting our daily operations significantly.",
            
            "Payment processing is timing out for all transactions. Customers are unable to complete purchases and we're losing revenue. The issue started this morning around 9 AM EST.",
            
            "Getting SSL certificate validation errors when trying to access the API from our production servers. The certificate appears to be valid but our requests are being rejected. Need urgent resolution.",
            
            "Our mobile app crashes immediately upon startup on both iOS and Android devices. This happened after the latest app store update. Users are unable to access any functionality.",
            
            "Database connections are timing out frequently, causing slow performance and failed operations. This is impacting user experience and system reliability. Need database optimization assistance.",
            
            "Email notifications have stopped sending completely. Users are not receiving password reset emails, welcome messages, or any automated communications. This is affecting user onboarding.",
            
            "Two-factor authentication is not working. Users receive SMS codes but the system says they're invalid. This is preventing secure access to accounts and causing security concerns."
        ]
        
        self.billing_descriptions = [
            "I was charged $299 instead of the usual $199 for my monthly subscription. I haven't made any plan changes and this appears to be an error. Please review my account and process a refund for the difference.",
            
            "My credit card expires next month and I need to update my payment method. However, the billing section shows an error when I try to save new payment information. Please help me update this before my next billing cycle.",
            
            "I was charged twice for the same monthly subscription on March 15th. My bank statement shows two identical charges of $199 each. I need one of these charges refunded immediately.",
            
            "I'm unable to download my invoice for February. The download link returns an error page. I need this invoice for my accounting records and expense reporting.",
            
            "I upgraded to the Enterprise plan last week and made the payment, but my account still shows the Basic plan features. The payment went through successfully but the upgrade wasn't applied.",
            
            "I need to change my billing cycle from monthly to annual to take advantage of the discount. However, I don't see this option in my account settings. Can you help me switch to annual billing?",
            
            "I received a notification that my payment failed, but my credit card is valid and has sufficient funds. I've contacted my bank and they confirm no issues on their end. Please investigate this payment failure.",
            
            "I signed up for the annual plan during your promotion to get 20% off, but my invoice shows the regular annual price. The discount wasn't applied and I should have saved $240.",
            
            "I need an official receipt for my company's accounting department. The email receipt I received doesn't have all the required tax information. Can you provide a proper invoice?",
            
            "My billing address changed and I need to update it for tax purposes. The current address on file is outdated and may cause issues with payment processing."
        ]
        
        self.product_descriptions = [
            "I'm new to the platform and need help setting up user permissions for my team. I have 15 team members with different roles and need to understand how to assign appropriate access levels to each person.",
            
            "Our team would benefit from training on the advanced features of your platform. We're currently using only basic functionality but want to leverage more powerful features to improve our workflow.",
            
            "I'd like to request an enhancement to the mobile app. It would be great to have offline capability so users can access their data even without an internet connection. This would significantly improve user experience.",
            
            "The API documentation is unclear regarding webhook integration. I'm a developer trying to set up real-time notifications but the examples provided don't match the actual API responses. Need clearer documentation.",
            
            "We need the ability to import large datasets in bulk. Currently, we have to manually enter each record which is time-consuming. A CSV or Excel import feature would save us hours of work daily.",
            
            "I need guidance on configuring automated workflows for our approval process. We want to set up automatic routing of requests based on amount thresholds and department rules.",
            
            "Can you provide best practices documentation for data organization and workflow optimization? We want to ensure we're using the platform efficiently and following recommended practices.",
            
            "Our new team members need comprehensive training materials. Do you have tutorials or training videos that cover all major features? We'd like to set up a structured onboarding process.",
            
            "I want to customize the dashboard layout to show only relevant widgets for different user types. Is there a way to create role-based dashboard configurations?",
            
            "We need to integrate your platform with our existing CRM system. Do you have pre-built connectors or APIs that would facilitate this integration? Need technical guidance on implementation."
        ]
    
    def generate_customers(self, count: int = 50) -> List[CustomerInfo]:
        """Generate sample customers."""
        customers = []
        
        for i in range(count):
            name = random.choice(self.customer_names)
            domain = random.choice(self.company_domains)
            email = f"{name.lower().replace(' ', '.')}.{random.randint(1, 999)}@{domain}"
            
            customer = CustomerInfo(
                id=str(uuid.uuid4()),
                name=name,
                email=email,
                phone=f"+1{random.randint(1000000000, 9999999999)}" if random.random() > 0.3 else None,
                sla_level=random.choices(
                    list(SLALevel),
                    weights=[50, 30, 15, 5],  # Basic, Standard, Premium, Enterprise
                    k=1
                )[0],
                account_type=random.choices(
                    ["free", "basic", "pro", "enterprise"],
                    weights=[20, 40, 30, 10],
                    k=1
                )[0],
                language="en",
                timezone=random.choice([
                    "America/New_York", "America/Los_Angeles", "Europe/London", 
                    "Europe/Berlin", "Asia/Tokyo", "Australia/Sydney"
                ])
            )
            
            customers.append(customer)
        
        return customers
    
    def generate_tickets(self, customers: List[CustomerInfo], count: int = 100) -> List[Ticket]:
        """Generate sample tickets."""
        tickets = []
        
        for i in range(count):
            customer = random.choice(customers)
            category = random.choices(
                list(TicketCategory),
                weights=[30, 25, 20, 10, 8, 5, 2],  # Technical, Billing, Product, etc.
                k=1
            )[0]
            
            # Select subject and description based on category
            if category == TicketCategory.TECHNICAL:
                subject = random.choice(self.technical_subjects)
                description = random.choice(self.technical_descriptions)
            elif category == TicketCategory.BILLING:
                subject = random.choice(self.billing_subjects)
                description = random.choice(self.billing_descriptions)
            else:  # Product, Account, etc.
                subject = random.choice(self.product_subjects)
                description = random.choice(self.product_descriptions)
            
            # Determine priority based on customer SLA and random factors
            if customer.sla_level == SLALevel.ENTERPRISE:
                priority = random.choices(
                    list(TicketPriority),
                    weights=[5, 15, 25, 30, 25],  # Favor higher priorities for enterprise
                    k=1
                )[0]
            elif customer.sla_level == SLALevel.PREMIUM:
                priority = random.choices(
                    list(TicketPriority),
                    weights=[10, 20, 30, 25, 15],
                    k=1
                )[0]
            else:
                priority = random.choices(
                    list(TicketPriority),
                    weights=[25, 35, 25, 10, 5],  # Lower priorities for basic/standard
                    k=1
                )[0]
            
            # Determine status with realistic distribution
            status = random.choices(
                list(TicketStatus),
                weights=[30, 25, 10, 20, 10, 5],  # Open, In Progress, Pending, Resolved, Closed, Escalated
                k=1
            )[0]
            
            # Create ticket with realistic timestamps
            days_ago = random.randint(1, 30)
            created_at = datetime.now(timezone.utc) - timedelta(days=days_ago)
            updated_at = created_at + timedelta(
                hours=random.randint(1, days_ago * 24)
            )
            
            ticket_id = f"TKT-{created_at.strftime('%Y%m%d')}-{i+1:04d}"
            
            ticket = Ticket(
                id=ticket_id,
                subject=subject,
                description=description,
                customer=customer,
                status=status,
                priority=priority,
                category=category,
                created_at=created_at,
                updated_at=updated_at,
                source=random.choice(["web", "email", "api", "chat"]),
                channel=random.choice(["support", "sales", "technical"])
            )
            
            # Add realistic comments
            self._add_sample_comments(ticket)
            
            # Set resolution/close times for completed tickets
            if status in [TicketStatus.RESOLVED, TicketStatus.CLOSED]:
                resolution_hours = random.randint(1, 72)
                ticket.resolved_at = ticket.created_at + timedelta(hours=resolution_hours)
                ticket.first_response_at = ticket.created_at + timedelta(
                    minutes=random.randint(15, 240)
                )
                
                if status == TicketStatus.CLOSED:
                    ticket.closed_at = ticket.resolved_at + timedelta(
                        hours=random.randint(1, 24)
                    )
            elif status == TicketStatus.IN_PROGRESS:
                ticket.first_response_at = ticket.created_at + timedelta(
                    minutes=random.randint(15, 120)
                )
            
            tickets.append(ticket)
        
        return tickets
    
    def _add_sample_comments(self, ticket: Ticket):
        """Add realistic comments to a ticket."""
        comment_count = random.randint(1, 5)
        
        for i in range(comment_count):
            hours_after_creation = random.randint(1, 48)
            
            if i == 0:  # First comment is usually from support
                author_type = "agent"
                author_id = "support-agent"
                content = "Thank you for contacting support. I've received your request and am investigating the issue. I'll get back to you shortly with an update."
            elif random.random() > 0.6:  # Customer response
                author_type = "customer"
                author_id = ticket.customer.id
                content = random.choice([
                    "Thank you for the quick response. I'm still experiencing the issue.",
                    "The suggested solution didn't work for me. Can you try something else?",
                    "I've tried the steps you mentioned but the problem persists.",
                    "This is working now, thank you for your help!",
                    "I need this resolved urgently as it's affecting my work.",
                    "Can you provide more detailed instructions?",
                    "I've followed your advice and it's partially working now."
                ])
            else:  # Agent follow-up
                author_type = "agent"
                author_id = f"agent-{random.randint(1, 5)}"
                content = random.choice([
                    "I've escalated this to our technical team for further investigation.",
                    "Please try the updated solution and let me know if it works.",
                    "I've identified the root cause and implemented a fix.",
                    "This appears to be a known issue. I'm working on a resolution.",
                    "I've updated your account settings. Please test and confirm.",
                    "The issue has been resolved on our end. Please verify on your side.",
                    "I need additional information to troubleshoot this further."
                ])
            
            ticket.add_comment(
                author_id=author_id,
                author_type=author_type,
                content=content,
                is_internal=random.random() > 0.8  # 20% chance of internal comment
            )
    
    def generate_knowledge_base_articles(self) -> List[Dict[str, Any]]:
        """Generate sample knowledge base articles."""
        articles = [
            {
                "id": "kb-login-issues",
                "title": "Troubleshooting Login Problems",
                "category": "authentication",
                "content": """
                Common login issues and their solutions:
                
                1. Forgot Password:
                   - Click 'Forgot Password' on the login page
                   - Enter your email address
                   - Check your inbox (and spam folder) for reset instructions
                   - Follow the link to create a new password
                
                2. Account Locked:
                   - Accounts are locked after 5 failed login attempts
                   - Wait 30 minutes for automatic unlock
                   - Contact support for immediate unlock if urgent
                
                3. Browser Issues:
                   - Clear your browser cache and cookies
                   - Try logging in using incognito/private mode
                   - Disable browser extensions that might interfere
                   - Update your browser to the latest version
                
                4. Two-Factor Authentication Problems:
                   - Ensure your device time is synchronized
                   - Try generating a new code
                   - Use backup codes if available
                   - Contact support if issues persist
                """,
                "tags": ["login", "password", "authentication", "2fa", "troubleshooting"],
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "rating": 4.8,
                "view_count": 2150,
                "helpful_votes": 189
            },
            {
                "id": "kb-api-integration",
                "title": "API Integration Best Practices",
                "category": "api",
                "content": """
                Guidelines for successful API integration:
                
                1. Authentication:
                   - Use Bearer token in Authorization header
                   - Format: 'Authorization: Bearer YOUR_API_KEY'
                   - Keep API keys secure and never expose in client-side code
                
                2. Rate Limiting:
                   - Free accounts: 100 requests/hour
                   - Pro accounts: 1,000 requests/hour
                   - Enterprise: 10,000 requests/hour
                   - Implement exponential backoff for retries
                
                3. Error Handling:
                   - Check HTTP status codes
                   - Parse error messages from response body
                   - Implement proper retry logic for temporary failures
                   - Log errors for debugging
                
                4. Best Practices:
                   - Use HTTPS for all requests
                   - Validate input data before sending
                   - Cache responses when appropriate
                   - Monitor API usage and performance
                """,
                "tags": ["api", "integration", "authentication", "rate-limits", "best-practices"],
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "rating": 4.6,
                "view_count": 1430,
                "helpful_votes": 156
            },
            {
                "id": "kb-billing-management",
                "title": "Managing Your Billing and Subscription",
                "category": "billing",
                "content": """
                Complete guide to billing and subscription management:
                
                1. Viewing Invoices:
                   - Go to Account Settings > Billing
                   - Click on any invoice to view details
                   - Download PDF receipts for your records
                
                2. Updating Payment Methods:
                   - Navigate to Billing > Payment Methods
                   - Add new credit card or PayPal account
                   - Set primary payment method for automatic billing
                
                3. Changing Plans:
                   - Visit Billing > Change Plan
                   - Select your desired plan
                   - Changes are prorated and take effect immediately
                
                4. Refund Policy:
                   - 30-day money-back guarantee for annual plans
                   - Prorated refunds for downgrades
                   - Contact support for refund requests
                
                5. Failed Payments:
                   - Update payment method within 7 days
                   - Service may be suspended after grace period
                   - Contact support for payment assistance
                """,
                "tags": ["billing", "subscription", "payment", "invoice", "refund"],
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "rating": 4.7,
                "view_count": 1820,
                "helpful_votes": 143
            }
        ]
        
        return articles
    
    def generate_sample_datasets(self, customers_count: int = 50, tickets_count: int = 100):
        """Generate and save all sample datasets."""
        print("Generating sample data...")
        
        # Create data directory if it doesn't exist
        os.makedirs("data/samples", exist_ok=True)
        
        # Generate customers
        print(f"Generating {customers_count} sample customers...")
        customers = self.generate_customers(customers_count)
        
        # Save customers
        customers_data = [
            {
                "id": c.id,
                "name": c.name,
                "email": c.email,
                "phone": c.phone,
                "sla_level": c.sla_level.value,
                "account_type": c.account_type,
                "language": c.language,
                "timezone": c.timezone,
                "metadata": c.metadata
            }
            for c in customers
        ]
        
        with open("data/samples/customers.json", "w") as f:
            json.dump(customers_data, f, indent=2)
        
        # Generate tickets
        print(f"Generating {tickets_count} sample tickets...")
        tickets = self.generate_tickets(customers, tickets_count)
        
        # Save tickets
        tickets_data = [ticket.to_dict() for ticket in tickets]
        
        with open("data/samples/tickets.json", "w") as f:
            json.dump(tickets_data, f, indent=2)
        
        # Generate knowledge base articles
        print("Generating knowledge base articles...")
        articles = self.generate_knowledge_base_articles()
        
        with open("data/samples/knowledge_base.json", "w") as f:
            json.dump(articles, f, indent=2)
        
        # Generate metrics and analytics data
        print("Generating metrics data...")
        metrics = self._generate_metrics_data(tickets)
        
        with open("data/samples/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Sample data generated successfully!")
        print(f"   📄 {len(customers)} customers saved to data/samples/customers.json")
        print(f"   🎫 {len(tickets)} tickets saved to data/samples/tickets.json")
        print(f"   📚 {len(articles)} articles saved to data/samples/knowledge_base.json")
        print(f"   📊 Metrics saved to data/samples/metrics.json")
        
        return {
            "customers": customers,
            "tickets": tickets,
            "articles": articles,
            "metrics": metrics
        }
    
    def _generate_metrics_data(self, tickets: List[Ticket]) -> Dict[str, Any]:
        """Generate sample metrics and analytics data."""
        
        # Calculate ticket distribution by category
        category_dist = {}
        for ticket in tickets:
            cat = ticket.category.value
            category_dist[cat] = category_dist.get(cat, 0) + 1
        
        # Calculate priority distribution
        priority_dist = {}
        for ticket in tickets:
            pri = ticket.priority.value
            priority_dist[pri] = priority_dist.get(pri, 0) + 1
        
        # Calculate status distribution
        status_dist = {}
        for ticket in tickets:
            stat = ticket.status.value
            status_dist[stat] = status_dist.get(stat, 0) + 1
        
        # Calculate SLA metrics
        resolved_tickets = [t for t in tickets if t.resolved_at]
        avg_resolution_time = 0
        if resolved_tickets:
            resolution_times = [
                (t.resolved_at - t.created_at).total_seconds() / 3600 
                for t in resolved_tickets
            ]
            avg_resolution_time = sum(resolution_times) / len(resolution_times)
        
        # Calculate customer satisfaction (simulated)
        satisfaction_scores = [random.uniform(3.5, 5.0) for _ in range(20)]
        avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores)
        
        return {
            "overview": {
                "total_tickets": len(tickets),
                "open_tickets": len([t for t in tickets if t.status != TicketStatus.CLOSED]),
                "resolved_tickets": len(resolved_tickets),
                "avg_resolution_time_hours": round(avg_resolution_time, 2),
                "customer_satisfaction": round(avg_satisfaction, 2)
            },
            "distributions": {
                "by_category": category_dist,
                "by_priority": priority_dist,
                "by_status": status_dist
            },
            "sla_metrics": {
                "first_response_sla": 92.5,
                "resolution_sla": 87.3,
                "escalation_rate": 8.2
            },
            "agent_performance": [
                {
                    "agent_id": "ticket-classifier-001",
                    "agent_name": "Ticket Classifier",
                    "tickets_processed": random.randint(80, 120),
                    "accuracy": random.uniform(0.85, 0.95),
                    "avg_processing_time": random.uniform(2.0, 5.0)
                },
                {
                    "agent_id": "technical-support-001",
                    "agent_name": "Technical Support",
                    "tickets_processed": random.randint(40, 60),
                    "resolution_rate": random.uniform(0.75, 0.90),
                    "avg_resolution_time": random.uniform(4.0, 8.0)
                },
                {
                    "agent_id": "billing-support-001",
                    "agent_name": "Billing Support",
                    "tickets_processed": random.randint(30, 50),
                    "resolution_rate": random.uniform(0.80, 0.95),
                    "avg_resolution_time": random.uniform(2.0, 6.0)
                }
            ],
            "generated_at": datetime.now(timezone.utc).isoformat()
        }


def main():
    """Main function to generate sample data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample data for AMP Support System")
    parser.add_argument("--customers", type=int, default=50, help="Number of customers to generate")
    parser.add_argument("--tickets", type=int, default=100, help="Number of tickets to generate")
    parser.add_argument("--output-dir", default="data/samples", help="Output directory for sample data")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate sample data
    generator = SampleDataGenerator()
    data = generator.generate_sample_datasets(args.customers, args.tickets)
    
    print(f"\n🎉 Sample data generation complete!")
    print(f"Files saved to: {args.output_dir}/")
    print(f"\nTo use this data in the support system:")
    print(f"1. Start the support system: python run_support_system.py")
    print(f"2. Access the web interface: http://localhost:8080")
    print(f"3. Import sample data through the admin interface")


if __name__ == "__main__":
    main()