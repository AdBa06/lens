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
from business_llm_validator import BusinessLLMValidator

logger = logging.getLogger(__name__)

class ClusterSummarizer:
    """Generate summaries for failure clusters using GPT-4"""
    
    def __init__(self):
        self.openai_client = self._create_openai_client()
    
    def _create_openai_client(self):
        """Create OpenAI client for regular OpenAI or Azure OpenAI"""
        if config.USE_AZURE_OPENAI:
            logger.info("Using Azure OpenAI")
            return openai.AzureOpenAI(
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                api_key=config.AZURE_OPENAI_API_KEY,
                api_version=config.AZURE_OPENAI_API_VERSION
            )
        elif config.OPENAI_API_KEY:
            logger.info("Using regular OpenAI")
            return openai.OpenAI(api_key=config.OPENAI_API_KEY)
        else:
            logger.warning("No OpenAI configuration found")
            return None
    
    def get_cluster_data(self, cluster_id: int) -> Dict[str, Any]:
        """Get comprehensive data for a cluster"""
        db = get_db_session()
        
        try:
            # Get cluster information
            cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
            if not cluster or cluster.is_noise:
                logger.warning(f"Cluster {cluster_id} not found or is noise")
                return {}
            
            # Get cluster assignments with related data
            assignments = db.query(ClusterAssignment).filter(
                ClusterAssignment.cluster_id == cluster_id
            ).join(Embedding).join(TelemetryEvent).all()
            
            # Collect cluster data
            customer_prompts = []
            skill_inputs = []
            skill_outputs = []
            ai_outputs = []
            fingerprints = []
            
            for assignment in assignments:
                event = assignment.embedding.event
                customer_prompts.append(event.customer_prompt)
                skill_inputs.append(event.skill_input)
                skill_outputs.append(event.skill_output)
                if event.ai_output:
                    ai_outputs.append(event.ai_output)
                
                # Get fingerprint data
                fingerprint = db.query(FailureFingerprint).filter(
                    FailureFingerprint.event_id == event.id
                ).first()
                if fingerprint:
                    fingerprints.append(fingerprint)
            
            # Extract common patterns
            plugins = [fp.plugin_name for fp in fingerprints if fp.plugin_name]
            endpoints = [fp.endpoint for fp in fingerprints if fp.endpoint]
            status_codes = [fp.status_code for fp in fingerprints if fp.status_code]
            error_types = [fp.error_type for fp in fingerprints if fp.error_type]
            error_messages = [fp.error_message for fp in fingerprints if fp.error_message]
            
            return {
                'cluster': cluster,
                'customer_prompts': customer_prompts,
                'skill_inputs': skill_inputs,
                'skill_outputs': skill_outputs,
                'ai_outputs': ai_outputs,
                'common_plugins': dict(Counter(plugins).most_common(10)),
                'common_endpoints': dict(Counter(endpoints).most_common(10)),
                'common_status_codes': dict(Counter(status_codes).most_common(10)),
                'common_error_types': dict(Counter(error_types).most_common(10)),
                'sample_error_messages': error_messages[:10]  # First 10 unique messages
            }
            
        except Exception as e:
            logger.error(f"Error getting cluster data for cluster {cluster_id}: {e}")
            return {}
        finally:
            db.close()
    
    def create_gpt_prompt(self, cluster_data: Dict[str, Any]) -> str:
        """Create prompt for GPT-4 analysis with dynamic, security-focused labeling"""
        cluster = cluster_data['cluster']
        sample_prompts = '\n'.join([f"- {prompt[:200]}..." for prompt in cluster_data['customer_prompts'][:5]])
        sample_errors = '\n'.join([f"- {error[:200]}..." for error in cluster_data['sample_error_messages'][:5]])
        sample_outputs = '\n'.join([f"- {json.dumps(output)[:300]}..." for output in cluster_data['skill_outputs'][:3]])

        prompt = f"""
You are an expert Security Copilot analyst. Given a cluster of Security Copilot prompts (both successful and failed), analyze what customers are trying to accomplish and what types of errors occur.

PROVIDE TWO PERSPECTIVES:

1. CUSTOMER INTENT ANALYSIS (for customer insights page):
- What are customers trying to accomplish with these prompts?
- What security topics/areas are they asking about?
- Generate a short, business-friendly label (max 4 words) describing the customer intent
- Write 2-3 sentences describing what customers in this cluster are trying to do

2. ERROR ANALYSIS (for error analysis page):
- What are the main technical failure patterns?
- What are the root causes of failures?
- What specific fixes should be implemented?

CLUSTER INFORMATION:
- Cluster ID: {cluster.id}
- Size: {cluster.size} events
- Algorithm: {cluster.cluster_algorithm}

COMMON PATTERNS:
- Plugins: {json.dumps(cluster_data['common_plugins'], indent=2)}
- Endpoints: {json.dumps(cluster_data['common_endpoints'], indent=2)}
- Status Codes: {json.dumps(cluster_data['common_status_codes'], indent=2)}
- Error Types: {json.dumps(cluster_data['common_error_types'], indent=2)}

SAMPLE CUSTOMER PROMPTS (first 5):
{sample_prompts}

SAMPLE ERROR MESSAGES (first 5):
{sample_errors}

SAMPLE SKILL OUTPUTS (first 3):
{sample_outputs}

IMPORTANT: You must respond with valid JSON only. Do not include any explanatory text before or after the JSON.

Format your response as JSON with the following structure:
{{
  "label": "Short customer intent label (max 4 words)",
  "customer_summary": "2-3 sentences describing what customers are trying to accomplish",
  "root_cause": "Technical root cause analysis of failures",
  "error_patterns": ["pattern1", "pattern2", "pattern3"],
  "recommendations": ["recommendation1", "recommendation2", "recommendation3"]
}}

Respond with ONLY the JSON object, no other text.
"""
        return prompt

    def generate_gpt_summary(self, prompt: str, cluster_id: int = None) -> Optional[Dict[str, Any]]:
        """Generate summary using GPT-4. Only accept valid JSON. If not valid JSON, try business validator as fallback. Always return a non-empty summary if possible."""
        if not self.openai_client:
            logger.warning("OpenAI client not configured")
            return None

        try:
            model_name = config.AZURE_OPENAI_DEPLOYMENT_NAME if config.USE_AZURE_OPENAI else config.OPENAI_MODEL
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing software failure patterns. Provide clear, actionable insights in the requested JSON format."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            content = response.choices[0].message.content

            # Try to parse JSON response
            import re
            def try_parse_json(text):
                try:
                    return json.loads(text)
                except Exception:
                    # Try to fix common JSON issues (e.g., missing closing bracket)
                    fixed = text.strip()
                    # Add missing closing bracket if it looks like a dict
                    if fixed.startswith('{') and not fixed.endswith('}'):
                        fixed += '}'
                        try:
                            return json.loads(fixed)
                        except Exception:
                            pass
                    return None

            summary = try_parse_json(content)
            if summary is None:
                # Try to extract JSON from markdown code block
                match = re.search(r"```json\\s*(\{.*?\})\\s*```", content, re.DOTALL)
                if match:
                    summary = try_parse_json(match.group(1))
                # Try to extract first JSON object anywhere in the text
                if summary is None:
                    match = re.search(r"(\{[\s\S]*?\})", content)
                    if match:
                        # Try to fix common JSON issues (like missing closing bracket)
                        json_str = match.group(1)
                        # Add missing closing bracket if needed
                        if json_str.count('{') > json_str.count('}'):
                            json_str = json_str + '}'
                        summary = try_parse_json(json_str)
            if summary is None:
                logger.warning(f"GPT response not valid JSON for cluster {cluster_id}. Raw response: {content}")
                summary = {
                    'label': '',
                    'customer_summary': '',
                    'root_cause': '',
                    'error_patterns': [],
                    'recommendations': []
                }

            # If summary is missing or not meaningful, try business validator
            if (not summary.get('label') or not summary.get('customer_summary')) and cluster_id is not None:
                try:
                    from business_llm_validator import BusinessLLMValidator
                    validator = BusinessLLMValidator()
                    business_summary = validator.validate_and_enhance_cluster(cluster_id)
                    # Map business summary fields to expected output
                    summary['label'] = business_summary.get('business_summary', '')[:50]
                    summary['customer_summary'] = business_summary.get('business_summary', '')
                    summary['root_cause'] = business_summary.get('root_cause', '')
                    summary['recommendations'] = business_summary.get('recommendations', [])
                except Exception as e:
                    logger.error(f"Business LLM validation failed: {e}")

            # Final fallback: if still empty, set to 'AI analysis not available'
            if not summary.get('label'):
                summary['label'] = 'AI analysis not available'
            if not summary.get('customer_summary'):
                summary['customer_summary'] = 'AI analysis not available'
            if not summary.get('root_cause'):
                summary['root_cause'] = ''
            if not summary.get('recommendations'):
                summary['recommendations'] = []
            return summary

        except Exception as e:
            logger.error(f"Error generating GPT summary: {e}")
            # Try business validator as last resort
            if cluster_id is not None:
                try:
                    from business_llm_validator import BusinessLLMValidator
                    validator = BusinessLLMValidator()
                    business_summary = validator.validate_and_enhance_cluster(cluster_id)
                    return {
                        'label': business_summary.get('business_summary', 'AI analysis not available')[:50],
                        'customer_summary': business_summary.get('business_summary', 'AI analysis not available'),
                        'root_cause': business_summary.get('root_cause', ''),
                        'error_patterns': [],
                        'recommendations': business_summary.get('recommendations', [])
                    }
                except Exception as e2:
                    logger.error(f"Business LLM validation failed: {e2}")
            return {
                'label': 'AI analysis not available',
                'customer_summary': 'AI analysis not available',
                'root_cause': '',
                'error_patterns': [],
                'recommendations': []
            }
    
    def _parse_text_response(self, content: str) -> Dict[str, Any]:
        """Parse non-JSON GPT response with better structure detection"""
        # Initialize defaults
        result = {
            'label': 'Customer Intent Analysis',
            'customer_summary': 'Customers are seeking security-related insights and analysis.',
            'root_cause': 'Analysis pending',
            'error_patterns': [],
            'recommendations': []
        }
        
        lines = content.split('\n')
        current_field = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for field indicators
            if '"label"' in line and ':' in line:
                # Extract label value
                label_part = line.split(':', 1)[1].strip().strip(',').strip('"')
                if label_part and label_part != 'null':
                    result['label'] = label_part
                    
            elif '"customer_summary"' in line and ':' in line:
                # Extract customer summary value  
                summary_part = line.split(':', 1)[1].strip().strip(',').strip('"')
                if summary_part and summary_part != 'null':
                    result['customer_summary'] = summary_part
                    current_field = 'customer_summary'
                    
            elif '"root_cause"' in line and ':' in line:
                # Extract root cause value
                cause_part = line.split(':', 1)[1].strip().strip(',').strip('"')
                if cause_part and cause_part != 'null':
                    result['root_cause'] = cause_part
                    current_field = 'root_cause'
                    
            elif current_field == 'customer_summary' and not line.startswith('"'):
                # Continue building customer summary
                result['customer_summary'] += ' ' + line.strip('"').strip(',')
                
            elif current_field == 'root_cause' and not line.startswith('"'):
                # Continue building root cause
                result['root_cause'] += ' ' + line.strip('"').strip(',')
        
        # Clean up the fields
        result['label'] = result['label'].strip('"').strip(',')[:50]  # Max 50 chars
        result['customer_summary'] = result['customer_summary'].strip('"').strip(',')
        result['root_cause'] = result['root_cause'].strip('"').strip(',')
        
        return result
    
    def summarize_cluster(self, cluster_id: int) -> bool:
        """Generate and store dynamic, security-focused summary and label for a cluster"""
        try:
            cluster_data = self.get_cluster_data(cluster_id)
            if not cluster_data:
                return False
            prompt = self.create_gpt_prompt(cluster_data)
            gpt_response = self.generate_gpt_summary(prompt, cluster_id=cluster_id)
            if not gpt_response:
                logger.error(f"Failed to generate GPT summary for cluster {cluster_id}")
                return False
            db = get_db_session()
            try:
                existing_summary = db.query(ClusterSummary).filter(
                    ClusterSummary.cluster_id == cluster_id
                ).first()
                if existing_summary:
                    db.delete(existing_summary)
                # Store LLM-generated label and summary
                cluster_summary = ClusterSummary(
                    cluster_id=cluster_id,
                    label=gpt_response.get('label', ''),
                    summary_text=gpt_response.get('customer_summary', ''),
                    root_cause=gpt_response.get('root_cause', ''),
                    recommendations=gpt_response.get('recommendations', []),
                    common_plugins=list(cluster_data['common_plugins'].keys()),
                    common_endpoints=list(cluster_data['common_endpoints'].keys()),
                    common_error_codes=list(cluster_data['common_status_codes'].keys()),
                    sample_prompts=cluster_data['customer_prompts'][:10],
                    gpt_model=config.OPENAI_MODEL,
                )
                db.add(cluster_summary)
                db.commit()
                logger.info(f"Successfully generated dynamic label and summary for cluster {cluster_id}")
                return True
            except Exception as e:
                logger.error(f"Error storing summary for cluster {cluster_id}: {e}")
                db.rollback()
                return False
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error summarizing cluster {cluster_id}: {e}")
            return False
    
    def summarize_all_clusters(self) -> int:
        """Generate summaries for all clusters"""
        db = get_db_session()
        
        try:
            # Get all non-noise clusters
            clusters = db.query(Cluster).filter(Cluster.is_noise == False).all()
            
            success_count = 0
            for cluster in clusters:
                if self.summarize_cluster(cluster.id):
                    success_count += 1
            
            logger.info(f"Successfully summarized {success_count}/{len(clusters)} clusters")
            return success_count
            
        except Exception as e:
            logger.error(f"Error in batch cluster summarization: {e}")
            return 0
        finally:
            db.close()
    
    def get_top_clusters(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top clusters by size with their summaries"""
        db = get_db_session()
        
        try:
            clusters = db.query(Cluster).filter(
                Cluster.is_noise == False
            ).order_by(Cluster.size.desc()).limit(limit).all()
            
            result = []
            for cluster in clusters:
                summary = db.query(ClusterSummary).filter(
                    ClusterSummary.cluster_id == cluster.id
                ).first()
                
                cluster_info = {
                    'cluster_id': cluster.id,
                    'size': cluster.size,
                    'algorithm': cluster.cluster_algorithm,
                    'summary': summary.summary_text if summary else None,
                    'root_cause': summary.root_cause if summary else None,
                    'common_plugins': summary.common_plugins if summary else [],
                    'common_endpoints': summary.common_endpoints if summary else [],
                    'common_error_codes': summary.common_error_codes if summary else [],
                    'sample_prompts': summary.sample_prompts if summary else []
                }
                
                result.append(cluster_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting top clusters: {e}")
            return []
        finally:
            db.close()

# Singleton instance
cluster_summarizer = ClusterSummarizer()