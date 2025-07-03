import openai
import json
import logging
from typing import Dict, Any, List, Optional
from collections import Counter
from sqlalchemy.orm import Session
from sqlalchemy import and_
from models import (
    Cluster, ClusterSummary, ClusterAssignment, 
    Embedding, TelemetryEvent, FailureFingerprint
)
from database import get_db_session
from config import config
from datetime import datetime

logger = logging.getLogger(__name__)

class BusinessLLMValidator:
    """LLM system for business validation and insights"""
    
    def __init__(self):
        self.openai_client = self._create_openai_client()
        self.business_context = {
            "company_focus": "Microsoft Office 365 and productivity tools",
            "user_personas": ["Business users", "IT administrators", "Security teams"],
            "critical_workflows": ["Authentication", "File sharing", "Email", "Calendar", "Teams collaboration"],
            "business_priorities": ["User productivity", "Security", "Compliance", "Cost efficiency"]
        }
    
    def _create_openai_client(self):
        """Create OpenAI client for regular OpenAI or Azure OpenAI"""
        if config.USE_AZURE_OPENAI:
            logger.info("Using Azure OpenAI for business validation")
            return openai.AzureOpenAI(
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                api_key=config.AZURE_OPENAI_API_KEY,
                api_version=config.AZURE_OPENAI_API_VERSION
            )
        elif config.OPENAI_API_KEY:
            logger.info("Using regular OpenAI for business validation")
            return openai.OpenAI(api_key=config.OPENAI_API_KEY)
        else:
            logger.warning("No OpenAI configuration found - using mock validation")
            return None
    
    def validate_and_enhance_cluster(self, cluster_id: int) -> Dict[str, Any]:
        """Validate cluster and provide business insights"""
        db = get_db_session()
        
        try:
            # Get comprehensive cluster data
            cluster_data = self._get_cluster_business_data(db, cluster_id)
            
            if not cluster_data:
                return {"error": "Cluster not found or empty"}
            
            # Generate business-focused analysis
            if self.openai_client:
                llm_analysis = self._generate_llm_business_analysis(cluster_data)
            else:
                llm_analysis = self._generate_mock_analysis(cluster_data)
            
            # Validate and enhance with business rules
            validated_analysis = self._apply_business_validation(cluster_data, llm_analysis)
            
            # Save to database
            self._save_business_summary(db, cluster_id, validated_analysis)
            
            return validated_analysis
            
        except Exception as e:
            logger.error(f"Error in business validation for cluster {cluster_id}: {e}")
            return {"error": str(e)}
        finally:
            db.close()
    
    def _get_cluster_business_data(self, db: Session, cluster_id: int) -> Dict[str, Any]:
        """Get comprehensive business-focused cluster data"""
        try:
            # Get cluster info
            cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
            if not cluster or cluster.is_noise:
                return {}
            
            # Get all events in cluster with their failure fingerprints
            events_query = db.query(TelemetryEvent, FailureFingerprint).join(
                Embedding, TelemetryEvent.id == Embedding.event_id
            ).join(
                ClusterAssignment, Embedding.id == ClusterAssignment.embedding_id
            ).outerjoin(
                FailureFingerprint, TelemetryEvent.id == FailureFingerprint.event_id
            ).filter(ClusterAssignment.cluster_id == cluster_id)
            
            events_data = events_query.all()
            
            if not events_data:
                return {}
            
            # Extract business-relevant information
            user_requests = []
            error_patterns = []
            business_impacts = []
            affected_services = set()
            error_types = Counter()
            time_patterns = []
            
            for event, fingerprint in events_data:
                # User request analysis
                if event.customer_prompt:
                    user_requests.append({
                        "prompt": event.customer_prompt,
                        "length": len(event.customer_prompt),
                        "intent": self._classify_user_intent(event.customer_prompt)
                    })
                    time_patterns.append(event.created_at)
                
                # Technical error analysis
                if fingerprint:
                    error_patterns.append({
                        "plugin": fingerprint.plugin_name,
                        "endpoint": fingerprint.endpoint,
                        "status_code": fingerprint.status_code,
                        "error_message": fingerprint.error_message,
                        "error_type": fingerprint.error_type
                    })
                    
                    if fingerprint.plugin_name:
                        affected_services.add(fingerprint.plugin_name)
                    if fingerprint.error_type:
                        error_types[fingerprint.error_type] += 1
                
                # Business impact assessment
                impact = self._assess_business_impact(event, fingerprint)
                business_impacts.append(impact)
            
            return {
                "cluster_id": cluster_id,
                "cluster_size": cluster.size,
                "user_requests": user_requests,
                "error_patterns": error_patterns,
                "business_impacts": business_impacts,
                "affected_services": list(affected_services),
                "error_type_distribution": dict(error_types),
                "time_patterns": time_patterns,
                "summary_stats": self._calculate_summary_stats(user_requests, error_patterns, business_impacts)
            }
            
        except Exception as e:
            logger.error(f"Error getting cluster business data: {e}")
            return {}
    
    def _classify_user_intent(self, prompt: str) -> str:
        """Classify user intent from prompt"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["meeting", "calendar", "schedule", "appointment"]):
            return "calendar_management"
        elif any(word in prompt_lower for word in ["file", "document", "share", "upload", "download"]):
            return "file_management"
        elif any(word in prompt_lower for word in ["email", "message", "send", "teams", "chat"]):
            return "communication"
        elif any(word in prompt_lower for word in ["data", "report", "analyze", "chart", "graph"]):
            return "data_analysis"
        elif any(word in prompt_lower for word in ["security", "audit", "signin", "access", "permission"]):
            return "security_admin"
        else:
            return "general_productivity"
    
    def _assess_business_impact(self, event: TelemetryEvent, fingerprint: FailureFingerprint) -> Dict[str, Any]:
        """Assess business impact of a failure"""
        impact = {
            "severity": "low",
            "user_productivity_impact": "minimal",
            "business_function_affected": "general",
            "estimated_user_frustration": "low"
        }
        
        # Analyze severity based on error patterns
        if fingerprint:
            if fingerprint.status_code in [500, 502, 503]:
                impact["severity"] = "high"
                impact["user_productivity_impact"] = "severe"
            elif fingerprint.status_code in [401, 403]:
                impact["severity"] = "medium"
                impact["user_productivity_impact"] = "moderate"
            
            # Service-specific impact
            if fingerprint.plugin_name:
                service = fingerprint.plugin_name.lower()
                if any(critical in service for critical in ["graph", "sharepoint", "teams", "outlook"]):
                    impact["business_function_affected"] = "critical_productivity"
                elif any(auth in service for auth in ["auth", "login", "signin"]):
                    impact["business_function_affected"] = "authentication"
        
        # User frustration based on prompt complexity
        if event.customer_prompt:
            if len(event.customer_prompt) > 100:  # Complex request
                impact["estimated_user_frustration"] = "high"
            elif len(event.customer_prompt) > 50:
                impact["estimated_user_frustration"] = "medium"
        
        return impact
    
    def _calculate_summary_stats(self, user_requests: List[Dict], error_patterns: List[Dict], business_impacts: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        return {
            "total_failures": len(user_requests),
            "avg_prompt_length": sum(r["length"] for r in user_requests) / len(user_requests) if user_requests else 0,
            "intent_distribution": Counter(r["intent"] for r in user_requests),
            "severity_distribution": Counter(i["severity"] for i in business_impacts),
            "most_common_errors": Counter(e.get("error_type") for e in error_patterns if e.get("error_type")).most_common(3),
            "affected_services_count": len(set(e.get("plugin") for e in error_patterns if e.get("plugin")))
        }
    
    def _generate_llm_business_analysis(self, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate LLM-powered business analysis"""
        try:
            # Create business-focused prompt
            prompt = self._create_business_analysis_prompt(cluster_data)
            
            # Get LLM response
            model_name = config.AZURE_OPENAI_DEPLOYMENT_NAME if config.USE_AZURE_OPENAI else config.OPENAI_MODEL
            
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": self._get_business_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse response
            analysis_text = response.choices[0].message.content
            
            # Structure the response
            return self._parse_llm_analysis(analysis_text, cluster_data)
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._generate_mock_analysis(cluster_data)
    
    def _get_business_system_prompt(self) -> str:
        """Get system prompt for business analysis"""
        return f"""You are a business analyst specializing in enterprise software failures and their impact on productivity.

CONTEXT:
- Company: Microsoft Office 365 ecosystem
- Users: {', '.join(self.business_context['user_personas'])}
- Critical Workflows: {', '.join(self.business_context['critical_workflows'])}
- Business Priorities: {', '.join(self.business_context['business_priorities'])}

INSTRUCTIONS:
1. Analyze the failure cluster from a business perspective
2. Focus on user productivity impact and business disruption
3. Provide actionable recommendations for business leaders
4. Categorize severity from business impact (not just technical)
5. Suggest prevention strategies that make business sense

FORMAT YOUR RESPONSE AS:
BUSINESS_SUMMARY: [2-3 sentences about what this cluster means for the business]
ROOT_CAUSE: [Business-friendly explanation of why this is happening]
USER_IMPACT: [How this affects end users and their work]
BUSINESS_RISK: [Potential business risks if not addressed]
RECOMMENDATIONS: [3-4 actionable items for business leaders]
PREVENTION: [How to prevent similar issues in the future]"""
    
    def _create_business_analysis_prompt(self, cluster_data: Dict[str, Any]) -> str:
        """Create business-focused analysis prompt"""
        stats = cluster_data["summary_stats"]
        
        prompt = f"""FAILURE CLUSTER ANALYSIS REQUEST

CLUSTER OVERVIEW:
- Cluster Size: {cluster_data['cluster_size']} failures
- Most Common User Intents: {dict(stats['intent_distribution'])}
- Severity Breakdown: {dict(stats['severity_distribution'])}
- Affected Services: {cluster_data['affected_services']}

SAMPLE USER REQUESTS:
{chr(10).join([f"- {req['prompt'][:100]}..." for req in cluster_data['user_requests'][:5]])}

TECHNICAL ERROR PATTERNS:
{chr(10).join([f"- {error.get('error_type', 'Unknown')}: {error.get('error_message', 'No message')[:80]}..." for error in cluster_data['error_patterns'][:5]])}

BUSINESS IMPACT INDICATORS:
- High Severity Failures: {stats['severity_distribution'].get('high', 0)}
- Authentication Issues: {sum(1 for r in cluster_data['user_requests'] if r['intent'] == 'security_admin')}
- Critical Service Failures: {len([s for s in cluster_data['affected_services'] if any(critical in s.lower() for critical in ['graph', 'sharepoint', 'teams', 'outlook'])])}

Please provide a comprehensive business analysis of this failure cluster."""
        
        return prompt
    
    def _parse_llm_analysis(self, analysis_text: str, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM analysis into structured format"""
        try:
            # Extract sections using simple parsing
            sections = {}
            current_section = None
            current_content = []
            
            for line in analysis_text.split('\n'):
                line = line.strip()
                if ':' in line and any(keyword in line.upper() for keyword in ['BUSINESS_SUMMARY', 'ROOT_CAUSE', 'USER_IMPACT', 'BUSINESS_RISK', 'RECOMMENDATIONS', 'PREVENTION']):
                    if current_section:
                        sections[current_section] = ' '.join(current_content)
                    current_section = line.split(':')[0].strip()
                    current_content = [line.split(':', 1)[1].strip()] if ':' in line else []
                elif current_section and line:
                    current_content.append(line)
            
            if current_section:
                sections[current_section] = ' '.join(current_content)
            
            # Structure the response
            return {
                "business_summary": sections.get("BUSINESS_SUMMARY", "Analysis not available"),
                "root_cause": sections.get("ROOT_CAUSE", "Root cause analysis not available"),
                "user_impact": sections.get("USER_IMPACT", "User impact assessment not available"),
                "business_risk": sections.get("BUSINESS_RISK", "Business risk not assessed"),
                "recommendations": sections.get("RECOMMENDATIONS", "No recommendations available").split('. '),
                "prevention_strategies": sections.get("PREVENTION", "No prevention strategies available").split('. '),
                "cluster_metadata": {
                    "size": cluster_data["cluster_size"],
                    "primary_intent": max(cluster_data["summary_stats"]["intent_distribution"], key=cluster_data["summary_stats"]["intent_distribution"].get) if cluster_data["summary_stats"]["intent_distribution"] else "unknown",
                    "severity_score": self._calculate_severity_score(cluster_data),
                    "business_priority": self._determine_business_priority(cluster_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing LLM analysis: {e}")
            return self._generate_mock_analysis(cluster_data)
    
    def _generate_mock_analysis(self, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock analysis when LLM is not available"""
        stats = cluster_data["summary_stats"]
        primary_intent = max(stats["intent_distribution"], key=stats["intent_distribution"].get) if stats["intent_distribution"] else "unknown"
        
        return {
            "business_summary": f"This cluster represents {cluster_data['cluster_size']} failures primarily affecting {primary_intent.replace('_', ' ')} workflows.",
            "root_cause": f"Technical failures in {', '.join(cluster_data['affected_services'][:3])} services causing user productivity disruption.",
            "user_impact": f"Users experiencing {stats['severity_distribution'].get('high', 0)} high-severity and {stats['severity_distribution'].get('medium', 0)} medium-severity failures.",
            "business_risk": "Potential productivity loss and user frustration if not addressed promptly.",
            "recommendations": [
                "Monitor affected services more closely",
                "Implement user communication strategy",
                "Review service dependencies",
                "Consider backup workflows"
            ],
            "prevention_strategies": [
                "Improve error handling in affected services",
                "Implement better service monitoring",
                "Provide user training on alternative workflows"
            ],
            "cluster_metadata": {
                "size": cluster_data["cluster_size"],
                "primary_intent": primary_intent,
                "severity_score": self._calculate_severity_score(cluster_data),
                "business_priority": self._determine_business_priority(cluster_data)
            }
        }
    
    def _calculate_severity_score(self, cluster_data: Dict[str, Any]) -> float:
        """Calculate business severity score (0-10)"""
        stats = cluster_data["summary_stats"]
        
        # Base score from severity distribution
        high_severity = stats["severity_distribution"].get("high", 0)
        medium_severity = stats["severity_distribution"].get("medium", 0)
        low_severity = stats["severity_distribution"].get("low", 0)
        total = high_severity + medium_severity + low_severity
        
        if total == 0:
            return 0
        
        severity_score = (high_severity * 10 + medium_severity * 5 + low_severity * 1) / total
        
        # Adjust for business-critical services
        critical_services = len([s for s in cluster_data['affected_services'] if any(critical in s.lower() for critical in ['graph', 'sharepoint', 'teams', 'outlook'])])
        if critical_services > 0:
            severity_score *= 1.5
        
        # Adjust for cluster size (larger clusters = more business impact)
        size_factor = min(2.0, 1 + (cluster_data['cluster_size'] - 10) / 50)
        severity_score *= size_factor
        
        return min(10.0, severity_score)
    
    def _determine_business_priority(self, cluster_data: Dict[str, Any]) -> str:
        """Determine business priority level"""
        severity_score = self._calculate_severity_score(cluster_data)
        
        if severity_score >= 8:
            return "Critical"
        elif severity_score >= 6:
            return "High"
        elif severity_score >= 4:
            return "Medium"
        else:
            return "Low"
    
    def _apply_business_validation(self, cluster_data: Dict[str, Any], llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply business validation rules to LLM analysis"""
        validated = llm_analysis.copy()
        
        # Validate severity score
        calculated_severity = self._calculate_severity_score(cluster_data)
        if abs(validated["cluster_metadata"]["severity_score"] - calculated_severity) > 2:
            validated["cluster_metadata"]["severity_score"] = calculated_severity
            validated["validation_notes"] = "Severity score adjusted based on business rules"
        
        # Validate business priority
        if validated["cluster_metadata"]["business_priority"] != self._determine_business_priority(cluster_data):
            validated["cluster_metadata"]["business_priority"] = self._determine_business_priority(cluster_data)
            validated["validation_notes"] = validated.get("validation_notes", "") + " Business priority recalculated"
        
        # Add validation timestamp
        validated["validation_timestamp"] = datetime.utcnow().isoformat()
        validated["validation_version"] = "1.0"
        
        return validated
    
    def _save_business_summary(self, db: Session, cluster_id: int, analysis: Dict[str, Any]):
        """Save business summary to database"""
        try:
            # Check if summary already exists
            existing_summary = db.query(ClusterSummary).filter(ClusterSummary.cluster_id == cluster_id).first()
            
            if existing_summary:
                # Update existing
                existing_summary.summary_text = analysis["business_summary"]
                existing_summary.root_cause = analysis["root_cause"]
                existing_summary.generated_at = datetime.utcnow()
            else:
                # Create new
                summary = ClusterSummary(
                    cluster_id=cluster_id,
                    summary_text=analysis["business_summary"],
                    root_cause=analysis["root_cause"],
                    gpt_model=config.OPENAI_MODEL,
                    generated_at=datetime.utcnow()
                )
                db.add(summary)
            
            db.commit()
            logger.info(f"Saved business summary for cluster {cluster_id}")
            
        except Exception as e:
            logger.error(f"Error saving business summary: {e}")
            db.rollback()

# Singleton instance
business_llm_validator = BusinessLLMValidator()
