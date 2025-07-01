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
        """Create prompt for GPT-4 analysis"""
        prompt = """Analyze the following cluster of failed Copilot responses and provide a comprehensive summary.

CLUSTER INFORMATION:
- Cluster ID: {cluster_id}
- Size: {size} failures
- Algorithm: {algorithm}

COMMON PATTERNS:
- Plugins: {plugins}
- Endpoints: {endpoints}
- Status Codes: {status_codes}
- Error Types: {error_types}

SAMPLE CUSTOMER PROMPTS (first 5):
{sample_prompts}

SAMPLE ERROR MESSAGES (first 5):
{sample_errors}

SAMPLE SKILL OUTPUTS (first 3):
{sample_outputs}

Please provide:
1. A clear, concise summary of what this cluster represents
2. The likely root cause of these failures
3. Common patterns and themes
4. Recommendations for resolution or investigation

Format your response as JSON with the following structure:
{{
    "summary": "Brief description of the cluster",
    "root_cause": "Primary reason for these failures",
    "patterns": ["pattern1", "pattern2", "pattern3"],
    "recommendations": ["recommendation1", "recommendation2"]
}}"""

        cluster = cluster_data['cluster']
        
        # Format sample data
        sample_prompts = '\n'.join([f"- {prompt[:200]}..." 
                                  for prompt in cluster_data['customer_prompts'][:5]])
        sample_errors = '\n'.join([f"- {error[:200]}..." 
                                 for error in cluster_data['sample_error_messages'][:5]])
        sample_outputs = '\n'.join([f"- {json.dumps(output)[:300]}..." 
                                  for output in cluster_data['skill_outputs'][:3]])
        
        return prompt.format(
            cluster_id=cluster.id,
            size=cluster.size,
            algorithm=cluster.cluster_algorithm,
            plugins=json.dumps(cluster_data['common_plugins'], indent=2),
            endpoints=json.dumps(cluster_data['common_endpoints'], indent=2),
            status_codes=json.dumps(cluster_data['common_status_codes'], indent=2),
            error_types=json.dumps(cluster_data['common_error_types'], indent=2),
            sample_prompts=sample_prompts,
            sample_errors=sample_errors,
            sample_outputs=sample_outputs
        )
    
    def generate_gpt_summary(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Generate summary using GPT-4"""
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
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Fallback to parsing structured text
                logger.warning("GPT response not valid JSON, attempting text parsing")
                return self._parse_text_response(content)
                
        except Exception as e:
            logger.error(f"Error generating GPT summary: {e}")
            return None
    
    def _parse_text_response(self, content: str) -> Dict[str, Any]:
        """Parse non-JSON GPT response"""
        # Simple fallback parsing
        lines = content.split('\n')
        
        summary = ""
        root_cause = ""
        patterns = []
        recommendations = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if 'summary' in line.lower():
                current_section = 'summary'
            elif 'root cause' in line.lower():
                current_section = 'root_cause'
            elif 'pattern' in line.lower():
                current_section = 'patterns'
            elif 'recommendation' in line.lower():
                current_section = 'recommendations'
            elif line.startswith('-') or line.startswith('•'):
                content_line = line[1:].strip()
                if current_section == 'patterns':
                    patterns.append(content_line)
                elif current_section == 'recommendations':
                    recommendations.append(content_line)
            else:
                if current_section == 'summary':
                    summary += line + " "
                elif current_section == 'root_cause':
                    root_cause += line + " "
        
        return {
            'summary': summary.strip(),
            'root_cause': root_cause.strip(),
            'patterns': patterns,
            'recommendations': recommendations
        }
    
    def summarize_cluster(self, cluster_id: int) -> bool:
        """Generate and store summary for a cluster"""
        try:
            # Get cluster data
            cluster_data = self.get_cluster_data(cluster_id)
            if not cluster_data:
                return False
            
            # Generate GPT prompt
            prompt = self.create_gpt_prompt(cluster_data)
            
            # Get GPT summary
            gpt_response = self.generate_gpt_summary(prompt)
            if not gpt_response:
                logger.error(f"Failed to generate GPT summary for cluster {cluster_id}")
                return False
            
            # Store summary in database
            db = get_db_session()
            
            try:
                # Remove existing summary
                existing_summary = db.query(ClusterSummary).filter(
                    ClusterSummary.cluster_id == cluster_id
                ).first()
                if existing_summary:
                    db.delete(existing_summary)
                
                # Create new summary
                cluster_summary = ClusterSummary(
                    cluster_id=cluster_id,
                    summary_text=gpt_response.get('summary', ''),
                    root_cause=gpt_response.get('root_cause', ''),
                    common_plugins=list(cluster_data['common_plugins'].keys()),
                    common_endpoints=list(cluster_data['common_endpoints'].keys()),
                    common_error_codes=list(cluster_data['common_status_codes'].keys()),
                    sample_prompts=cluster_data['customer_prompts'][:10],
                    gpt_model=config.OPENAI_MODEL
                )
                
                db.add(cluster_summary)
                db.commit()
                
                logger.info(f"Successfully generated summary for cluster {cluster_id}")
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