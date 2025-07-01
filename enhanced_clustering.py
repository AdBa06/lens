#!/usr/bin/env python3
"""
Enhanced Clustering for Copilot Failures
Hybrid approach: Domain-guided clustering with LLM validation and business context
"""

import numpy as np
import logging
import json
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import openai

from models import Embedding, Cluster, ClusterSummary, ClusterAssignment, TelemetryEvent, FailureFingerprint
from database import get_db_session
from config import config

logger = logging.getLogger(__name__)

# Copilot Business Taxonomy
COPILOT_TAXONOMY = {
    "graph_api_issues": {
        "name": "Graph API Query Issues",
        "keywords": ["query", "parameter", "odata", "filter", "select", "expand", "count", "parsing"],
        "error_codes": [400, 404],
        "plugins": ["Graph", "Calendar", "Email", "Files"],
        "fix_team": "Graph API Team",
        "priority": "High",
        "typical_fix_time": "2-3 weeks",
        "business_impact": "High - Users can't query data"
    },
    "authentication_issues": {
        "name": "Authentication & Permission Issues", 
        "keywords": ["auth", "permission", "forbidden", "unauthorized", "scope", "token"],
        "error_codes": [401, 403],
        "plugins": ["Calendar", "Email", "Files", "Teams"],
        "fix_team": "Identity Team",
        "priority": "Critical",
        "typical_fix_time": "1-2 days",
        "business_impact": "Critical - Users can't access resources"
    },
    "template_issues": {
        "name": "Template & Variable Issues",
        "keywords": ["placeholder", "variable", "<", ">", "unresolved", "template", "VARIABLE_NAME"],
        "error_codes": [500],
        "plugins": ["All"],
        "fix_team": "Copilot Core Team", 
        "priority": "Critical",
        "typical_fix_time": "1 week",
        "business_impact": "Critical - Core Copilot functionality broken"
    },
    "api_infrastructure": {
        "name": "API Infrastructure Issues",
        "keywords": ["timeout", "server", "internal", "unavailable", "rate limit", "throttl"],
        "error_codes": [500, 502, 503, 504, 429],
        "plugins": ["All"],
        "fix_team": "Platform Team",
        "priority": "High", 
        "typical_fix_time": "3-5 days",
        "business_impact": "High - Service reliability affected"
    },
    "user_input_issues": {
        "name": "User Input & Intent Issues",
        "keywords": ["invalid", "malformed", "unsupported", "syntax", "format"],
        "error_codes": [400],
        "plugins": ["All"],
        "fix_team": "Copilot UX Team",
        "priority": "Medium",
        "typical_fix_time": "2-4 weeks", 
        "business_impact": "Medium - User experience degraded"
    }
}

class EnhancedClusteringEngine:
    """Enhanced clustering with domain knowledge and LLM validation"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.openai_client = self._create_openai_client() if config.OPENAI_API_KEY else None
        
    def _create_openai_client(self):
        """Create OpenAI client"""
        try:
            if config.USE_AZURE_OPENAI:
                return openai.AzureOpenAI(
                    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                    api_key=config.AZURE_OPENAI_API_KEY,
                    api_version=config.AZURE_OPENAI_API_VERSION
                )
            else:
                return openai.OpenAI(api_key=config.OPENAI_API_KEY)
        except Exception as e:
            logger.error(f"Error creating OpenAI client: {e}")
            return None
    
    def classify_failure_by_taxonomy(self, event_data: Dict[str, Any]) -> Optional[str]:
        """Classify a failure into taxonomy categories"""
        skill_output = event_data.get('skill_output', {})
        customer_prompt = event_data.get('customer_prompt', '')
        
        # Extract error info
        error_message = str(skill_output).lower()
        status_code = skill_output.get('status_code') if isinstance(skill_output, dict) else None
        
        # Score each taxonomy category
        scores = {}
        for tax_key, tax_info in COPILOT_TAXONOMY.items():
            score = 0
            
            # Keyword matching
            for keyword in tax_info['keywords']:
                if keyword.lower() in error_message or keyword.lower() in customer_prompt.lower():
                    score += 2
            
            # Error code matching
            if status_code and status_code in tax_info['error_codes']:
                score += 3
                
            scores[tax_key] = score
        
        # Return category with highest score (if > 0)
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return None
    
    def get_taxonomy_guided_clusters(self, embeddings_data: List[Dict]) -> Dict[str, List[int]]:
        """Group failures by taxonomy categories first"""
        taxonomy_groups = {tax_key: [] for tax_key in COPILOT_TAXONOMY.keys()}
        taxonomy_groups['uncategorized'] = []
        
        for i, data in enumerate(embeddings_data):
            category = self.classify_failure_by_taxonomy(data)
            if category:
                taxonomy_groups[category].append(i)
            else:
                taxonomy_groups['uncategorized'].append(i)
                
        # Remove empty groups
        return {k: v for k, v in taxonomy_groups.items() if v}
    
    def perform_enhanced_clustering(self, embeddings: np.ndarray, 
                                  embedding_data: List[Dict]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform taxonomy-guided clustering with validation"""
        
        # Step 1: Get taxonomy-based initial groupings
        logger.info("🎯 Step 1: Applying business taxonomy...")
        taxonomy_groups = self.get_taxonomy_guided_clusters(embedding_data)
        
        initial_clusters = 0
        cluster_labels = np.full(len(embeddings), -1)
        
        # Assign initial taxonomy labels
        for tax_key, indices in taxonomy_groups.items():
            if len(indices) >= 3:  # Minimum cluster size
                for idx in indices:
                    cluster_labels[idx] = initial_clusters
                initial_clusters += 1
                logger.info(f"📋 {COPILOT_TAXONOMY.get(tax_key, {}).get('name', tax_key)}: {len(indices)} failures")
        
        # Step 2: Sub-cluster large taxonomy groups using K-means
        logger.info("🔄 Step 2: Refining large clusters...")
        refined_labels = cluster_labels.copy()
        next_cluster_id = initial_clusters
        
        for tax_key, indices in taxonomy_groups.items():
            if len(indices) > 15:  # Large groups need sub-clustering
                sub_embeddings = embeddings[indices]
                sub_embeddings_scaled = self.scaler.fit_transform(sub_embeddings)
                
                # Determine optimal sub-clusters
                n_subclusters = min(max(2, len(indices) // 8), 4)
                
                kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
                sub_labels = kmeans.fit_predict(sub_embeddings_scaled)
                
                # Reassign labels
                for i, orig_idx in enumerate(indices):
                    refined_labels[orig_idx] = next_cluster_id + sub_labels[i]
                
                next_cluster_id += n_subclusters
                logger.info(f"🔍 Split {tax_key} into {n_subclusters} sub-clusters")
        
        # Step 3: LLM validation for problematic clusters
        logger.info("🤖 Step 3: LLM validation...")
        validated_labels = self._llm_validate_clusters(refined_labels, embedding_data)
        
        # Calculate metrics
        unique_labels = set(validated_labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        n_noise = list(validated_labels).count(-1)
        
        # Calculate silhouette score (exclude noise points)
        non_noise_mask = validated_labels != -1
        silhouette_avg = 0
        if len(set(validated_labels[non_noise_mask])) > 1:
            try:
                silhouette_avg = silhouette_score(embeddings[non_noise_mask], validated_labels[non_noise_mask])
            except:
                pass
        
        metrics = {
            'algorithm': 'enhanced_hybrid',
            'taxonomy_groups': len(taxonomy_groups),
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': silhouette_avg,
            'taxonomy_mapping': taxonomy_groups
        }
        
        logger.info(f"✅ Enhanced clustering completed: {n_clusters} clusters, {n_noise} noise points")
        return validated_labels, metrics
    
    def _llm_validate_clusters(self, cluster_labels: np.ndarray, 
                             embedding_data: List[Dict]) -> np.ndarray:
        """Advanced LLM validation with business intelligence"""
        if not self.openai_client:
            logger.warning("⚠️ No OpenAI client available, skipping LLM validation")
            return cluster_labels
        
        logger.info("🤖 Step 3a: Running advanced cluster coherence analysis...")
        validated_labels = cluster_labels.copy()
        unique_labels = [l for l in set(cluster_labels) if l != -1]
        
        # Phase 1: Validate cluster coherence
        for cluster_id in unique_labels:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            # Validate clusters with sufficient size
            if len(cluster_indices) >= 3:
                validation_result = self._advanced_cluster_validation(cluster_id, cluster_indices, embedding_data)
                
                if validation_result:
                    if validation_result.get('should_split'):
                        validated_labels = self._intelligent_cluster_split(
                            validated_labels, cluster_id, cluster_indices, 
                            embedding_data, validation_result
                        )
                    elif validation_result.get('should_merge_with'):
                        # Note potential merge candidates for later
                        logger.info(f"📝 Cluster {cluster_id} flagged for potential merge")
        
        # Phase 2: Cross-cluster pattern detection
        logger.info("🕸️ Step 3b: Detecting cross-cluster patterns...")
        self._detect_cross_cluster_patterns(validated_labels, embedding_data)
        
        return validated_labels
    
    def _advanced_cluster_validation(self, cluster_id: int, indices: np.ndarray, 
                                   embedding_data: List[Dict]) -> Optional[Dict]:
        """Advanced cluster validation with business intelligence"""
        try:
            # Sample failures strategically (not just random)
            sample_size = min(8, len(indices))
            if len(indices) <= 8:
                sample_indices = indices
            else:
                # Take diverse samples: some from beginning, middle, end
                step = len(indices) // sample_size
                sample_indices = indices[::step][:sample_size]
            
            samples = []
            error_patterns = {}
            
            for idx in sample_indices:
                data = embedding_data[idx]
                error_str = str(data.get('skill_output', {}))
                
                # Extract error patterns
                if 'status_code' in error_str:
                    status_match = error_str
                    if '400' in status_match:
                        error_patterns['400_errors'] = error_patterns.get('400_errors', 0) + 1
                    elif '401' in status_match or '403' in status_match:
                        error_patterns['auth_errors'] = error_patterns.get('auth_errors', 0) + 1
                    elif '500' in status_match:
                        error_patterns['server_errors'] = error_patterns.get('server_errors', 0) + 1
                
                samples.append({
                    'prompt': data.get('customer_prompt', '')[:300],
                    'error': error_str[:300],
                    'skill': data.get('skill_input', '')
                })
            
            prompt = f"""
You are an expert failure analysis engineer. Analyze this cluster of {len(indices)} Copilot failures.

Sample failures:
{json.dumps(samples, indent=2)}

Error pattern summary: {error_patterns}

Analyze for:
1. ROOT CAUSE COHERENCE: Do these failures share the same underlying root cause?
2. TEAM OWNERSHIP: Would the same engineering team handle all these failures?
3. FIX STRATEGY: Can these be fixed with the same type of solution?
4. BUSINESS IMPACT: Do they affect users in the same way?
5. TEMPORAL PATTERNS: Could these be cascading failures?

Critical questions:
- Are there 2+ distinct root causes mixed together?
- Should this be split into sub-problems?
- Are there urgent vs non-urgent issues mixed?
- Is this actually multiple failure modes?

Respond with JSON:
{{
    "coherence_score": 0.0-1.0,
    "should_split": true/false,
    "split_reasoning": "specific technical reason",
    "business_priority": "Critical/High/Medium/Low",
    "escalation_needed": true/false,
    "suggested_split_criteria": "how to split if needed",
    "cross_cluster_patterns": ["pattern1", "pattern2"]
}}
"""
            
            model_name = config.AZURE_OPENAI_DEPLOYMENT_NAME if config.USE_AZURE_OPENAI else config.OPENAI_MODEL
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a senior software failure analysis expert with deep knowledge of system reliability patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            response_content = response.choices[0].message.content.strip()
            logger.debug(f"Raw LLM response: {response_content}")
            
            # Try to parse JSON, with fallback for malformed responses
            try:
                result = json.loads(response_content)
            except json.JSONDecodeError:
                # Try to extract JSON from response if it's wrapped in text
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse LLM JSON response for cluster {cluster_id}, using defaults")
                        result = {
                            "coherence_score": 0.7,  # Assume moderate coherence
                            "should_split": False,
                            "business_priority": "Medium",
                            "escalation_needed": False
                        }
                else:
                    logger.warning(f"No JSON found in LLM response for cluster {cluster_id}, using defaults")
                    result = {
                        "coherence_score": 0.7,
                        "should_split": False, 
                        "business_priority": "Medium",
                        "escalation_needed": False
                    }
            
            logger.info(f"🔍 Cluster {cluster_id} coherence: {result.get('coherence_score', 'N/A')}")
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced cluster validation: {e}")
            # Return sensible defaults instead of None
            return {
                "coherence_score": 0.5,
                "should_split": False,
                "business_priority": "Medium", 
                "escalation_needed": False
            }
    
    def _intelligent_cluster_split(self, cluster_labels: np.ndarray, cluster_id: int, 
                                 cluster_indices: np.ndarray, embedding_data: List[Dict],
                                 validation_result: Dict) -> np.ndarray:
        """Intelligently split clusters based on LLM analysis"""
        try:
            split_criteria = validation_result.get('suggested_split_criteria', '')
            new_labels = cluster_labels.copy()
            
            # Smart splitting based on LLM suggestions
            if 'auth' in split_criteria.lower() or 'permission' in split_criteria.lower():
                # Split by authentication vs other errors
                auth_indices = []
                other_indices = []
                
                for idx in cluster_indices:
                    data = embedding_data[idx]
                    error_str = str(data.get('skill_output', {})).lower()
                    if any(term in error_str for term in ['auth', 'forbidden', 'unauthorized', '401', '403']):
                        auth_indices.append(idx)
                    else:
                        other_indices.append(idx)
                
                if len(auth_indices) >= 2 and len(other_indices) >= 2:
                    new_cluster_id = max(cluster_labels) + 1
                    for idx in auth_indices:
                        new_labels[idx] = new_cluster_id
                    logger.info(f"🔧 Split cluster {cluster_id} by auth vs non-auth errors")
            
            elif 'status' in split_criteria.lower() or 'error code' in split_criteria.lower():
                # Split by error codes
                client_errors = []  # 4xx
                server_errors = []  # 5xx
                
                for idx in cluster_indices:
                    data = embedding_data[idx]
                    error_str = str(data.get('skill_output', {}))
                    if any(code in error_str for code in ['400', '404', '422']):
                        client_errors.append(idx)
                    elif any(code in error_str for code in ['500', '502', '503']):
                        server_errors.append(idx)
                
                if len(client_errors) >= 2 and len(server_errors) >= 2:
                    new_cluster_id = max(cluster_labels) + 1
                    for idx in server_errors:
                        new_labels[idx] = new_cluster_id
                    logger.info(f"🔧 Split cluster {cluster_id} by client vs server errors")
            
            else:
                # Default: split by first vs second half (fallback)
                mid_point = len(cluster_indices) // 2
                new_cluster_id = max(cluster_labels) + 1
                for i in cluster_indices[mid_point:]:
                    new_labels[i] = new_cluster_id
                logger.info(f"🔧 Split cluster {cluster_id} using default strategy")
            
            return new_labels
            
        except Exception as e:
            logger.error(f"Error in intelligent cluster splitting: {e}")
            return cluster_labels
    
    def _detect_cross_cluster_patterns(self, cluster_labels: np.ndarray, embedding_data: List[Dict]):
        """Detect patterns across multiple clusters"""
        try:
            unique_labels = [l for l in set(cluster_labels) if l != -1]
            
            # Temporal pattern detection
            temporal_patterns = {}
            error_cascades = []
            
            for label in unique_labels:
                cluster_indices = np.where(cluster_labels == label)[0]
                cluster_data = [embedding_data[i] for i in cluster_indices]
                
                # Check for common error sequences
                error_types = []
                for data in cluster_data:
                    error_str = str(data.get('skill_output', {}))
                    if 'timeout' in error_str.lower():
                        error_types.append('timeout')
                    elif any(code in error_str for code in ['500', '502', '503']):
                        error_types.append('server_error')
                    elif any(code in error_str for code in ['401', '403']):
                        error_types.append('auth_error')
                
                if error_types:
                    most_common = Counter(error_types).most_common(1)[0][0]
                    temporal_patterns[label] = most_common
            
            # Detect potential cascading failures
            if 'timeout' in temporal_patterns.values() and 'server_error' in temporal_patterns.values():
                logger.info("⚠️ Potential cascading failure detected: timeouts leading to server errors")
                
            # Log insights
            if temporal_patterns:
                logger.info(f"🔍 Cross-cluster patterns detected: {temporal_patterns}")
                
        except Exception as e:
            logger.error(f"Error in cross-cluster pattern detection: {e}")

class EnhancedClusterManager:
    """Manage enhanced clusters with business context"""
    
    def __init__(self):
        self.clustering_engine = EnhancedClusteringEngine()
    
    def perform_enhanced_clustering(self) -> bool:
        """Perform the complete enhanced clustering pipeline"""
        try:
            logger.info("🚀 Starting Enhanced Clustering Pipeline...")
            
            # Get embeddings and related data
            db = get_db_session()
            embeddings_query = db.query(Embedding).join(TelemetryEvent).all()
            
            if not embeddings_query:
                logger.warning("No embeddings found")
                return False
            
            # Prepare data
            vectors = []
            embedding_data = []
            embedding_ids = []
            
            for emb in embeddings_query:
                vectors.append(emb.embedding_vector)
                embedding_ids.append(emb.id)
                
                # Get event data for taxonomy classification
                event_data = {
                    'customer_prompt': emb.event.customer_prompt,
                    'skill_input': emb.event.skill_input,
                    'skill_output': emb.event.skill_output
                }
                embedding_data.append(event_data)
            
            db.close()
            
            vectors_array = np.array(vectors)
            
            # Perform enhanced clustering
            cluster_labels, metrics = self.clustering_engine.perform_enhanced_clustering(
                vectors_array, embedding_data
            )
            
            # Store results
            success = self._store_enhanced_results(embedding_ids, cluster_labels, metrics)
            
            if success:
                logger.info("🎉 Enhanced clustering pipeline completed successfully!")
                
                # Generate enhanced summaries
                self._generate_enhanced_summaries()
                
            return success
            
        except Exception as e:
            logger.error(f"Error in enhanced clustering: {e}")
            return False
    
    def _store_enhanced_results(self, embedding_ids: List[int], 
                              cluster_labels: np.ndarray, 
                              metrics: Dict[str, Any]) -> bool:
        """Store enhanced clustering results"""
        db = get_db_session()
        
        try:
            # Clear existing results
            db.query(ClusterAssignment).delete()
            db.query(ClusterSummary).delete()
            db.query(Cluster).delete()
            db.commit()
            
            # Create enhanced cluster records
            unique_labels = set(cluster_labels)
            cluster_map = {}
            
            for label in unique_labels:
                cluster_size = list(cluster_labels).count(label)
                is_noise = (label == -1)
                
                # Find taxonomy category for this cluster
                taxonomy_category = self._identify_cluster_taxonomy(label, embedding_ids, cluster_labels, db)
                
                cluster = Cluster(
                    cluster_label=int(label),
                    cluster_algorithm='enhanced_hybrid',
                    cluster_parameters={
                        **metrics,
                        'taxonomy_category': taxonomy_category
                    },
                    size=cluster_size,
                    is_noise=is_noise
                )
                
                db.add(cluster)
                db.flush()
                cluster_map[label] = cluster.id
            
            # Create assignments
            for i, (embedding_id, label) in enumerate(zip(embedding_ids, cluster_labels)):
                assignment = ClusterAssignment(
                    embedding_id=embedding_id,
                    cluster_id=cluster_map[label]
                )
                db.add(assignment)
            
            db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error storing enhanced results: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def _identify_cluster_taxonomy(self, cluster_label: int, embedding_ids: List[int],
                                 cluster_labels: np.ndarray, db) -> Optional[str]:
        """Identify which taxonomy category this cluster belongs to"""
        cluster_indices = np.where(cluster_labels == cluster_label)[0]
        
        if len(cluster_indices) == 0:
            return None
        
        # Get sample events from this cluster
        sample_embeddings = [embedding_ids[i] for i in cluster_indices[:5]]
        events = db.query(TelemetryEvent).join(Embedding).filter(
            Embedding.id.in_(sample_embeddings)
        ).all()
        
        # Score taxonomy categories
        category_scores = {}
        for event in events:
            event_data = {
                'customer_prompt': event.customer_prompt,
                'skill_output': event.skill_output
            }
            category = self.clustering_engine.classify_failure_by_taxonomy(event_data)
            if category:
                category_scores[category] = category_scores.get(category, 0) + 1
        
        # Return most common category
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return None
    
    def _generate_enhanced_summaries(self):
        """Generate enhanced summaries with business intelligence"""
        from summarization import cluster_summarizer
        
        # Generate AI summaries
        summary_count = cluster_summarizer.summarize_all_clusters()
        
        # Add business context to summaries
        self._add_business_context_to_summaries()
        
        # Add dynamic priority scoring
        self._add_dynamic_priority_scoring()
        
        # Generate escalation recommendations
        self._generate_escalation_recommendations()
        
        logger.info(f"📝 Generated {summary_count} enhanced summaries with business intelligence")
    
    def _add_business_context_to_summaries(self):
        """Add business context to existing summaries"""
        db = get_db_session()
        
        try:
            clusters = db.query(Cluster).all()
            
            for cluster in clusters:
                # Get taxonomy category
                taxonomy_category = cluster.cluster_parameters.get('taxonomy_category')
                
                if taxonomy_category and taxonomy_category in COPILOT_TAXONOMY:
                    tax_info = COPILOT_TAXONOMY[taxonomy_category]
                    
                    # Update or create summary with business context
                    summary = db.query(ClusterSummary).filter(
                        ClusterSummary.cluster_id == cluster.id
                    ).first()
                    
                    if summary:
                        # Enhance existing summary
                        business_context = f"\n\n🏢 BUSINESS CONTEXT:\n"
                        business_context += f"• Responsible Team: {tax_info['fix_team']}\n"
                        business_context += f"• Priority: {tax_info['priority']}\n"
                        business_context += f"• Typical Fix Time: {tax_info['typical_fix_time']}\n"
                        business_context += f"• Business Impact: {tax_info['business_impact']}"
                        
                        summary.summary_text = (summary.summary_text or "") + business_context
                        summary.root_cause = (summary.root_cause or "") + f"\n\nCategory: {tax_info['name']}"
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error adding business context: {e}")
            db.rollback()
        finally:
            db.close()
    
    def _add_dynamic_priority_scoring(self):
        """Add dynamic priority scoring based on trends and business impact"""
        try:
            logger.info("🎯 Adding dynamic priority scoring...")
            db = get_db_session()
            clusters = db.query(Cluster).filter(Cluster.cluster_algorithm == 'enhanced_hybrid').all()
            
            for cluster in clusters:
                # Base priority from taxonomy
                taxonomy_category = cluster.cluster_parameters.get('taxonomy_category')
                base_priority = 2  # Medium default
                
                if taxonomy_category in COPILOT_TAXONOMY:
                    taxonomy_priority = COPILOT_TAXONOMY[taxonomy_category].get('priority', 'Medium')
                    if taxonomy_priority == 'Critical':
                        base_priority = 4
                    elif taxonomy_priority == 'High':
                        base_priority = 3
                    elif taxonomy_priority == 'Low':
                        base_priority = 1
                
                # Size multiplier (larger clusters = more impact)
                size_multiplier = min(1.5, 1 + (cluster.size - 5) * 0.1) if cluster.size > 5 else 1.0
                
                # Calculate dynamic score
                dynamic_score = base_priority * size_multiplier
                
                # Update cluster parameters with scoring
                if not cluster.cluster_parameters:
                    cluster.cluster_parameters = {}
                
                # Create new dict to force SQLAlchemy to detect changes
                new_params = dict(cluster.cluster_parameters)
                new_params.update({
                    'dynamic_priority_score': round(dynamic_score, 2),
                    'base_priority': base_priority,
                    'size_multiplier': round(size_multiplier, 2),
                    'computed_priority': 'Critical' if dynamic_score >= 4.0 else 
                                       'High' if dynamic_score >= 3.0 else 
                                       'Medium' if dynamic_score >= 2.0 else 'Low'
                })
                cluster.cluster_parameters = new_params
                
                # Mark for update
                db.add(cluster)
            
            db.commit()
            db.close()
            logger.info("✅ Dynamic priority scoring completed")
            
        except Exception as e:
            logger.error(f"Error in dynamic priority scoring: {e}")
    
    def _generate_escalation_recommendations(self):
        """Generate smart escalation recommendations for critical clusters"""
        try:
            logger.info("🚨 Generating escalation recommendations...")
            db = get_db_session()
            clusters = db.query(Cluster).filter(Cluster.cluster_algorithm == 'enhanced_hybrid').all()
            
            escalation_clusters = []
            
            for cluster in clusters:
                dynamic_score = cluster.cluster_parameters.get('dynamic_priority_score', 0)
                taxonomy_category = cluster.cluster_parameters.get('taxonomy_category')
                
                # Escalation criteria
                needs_escalation = False
                escalation_reasons = []
                
                # High priority + large size
                if dynamic_score >= 3.5 and cluster.size >= 10:
                    needs_escalation = True
                    escalation_reasons.append(f"High impact failure affecting {cluster.size} users")
                
                # Critical business functions
                if taxonomy_category in ['authentication_issues', 'template_issues']:
                    needs_escalation = True
                    escalation_reasons.append("Critical business function impacted")
                
                # Large clusters (potential widespread issue)
                if cluster.size >= 15:
                    needs_escalation = True
                    escalation_reasons.append("Widespread failure pattern detected")
                
                if needs_escalation:
                    escalation_clusters.append({
                        'cluster_id': cluster.id,
                        'size': cluster.size,
                        'category': taxonomy_category,
                        'dynamic_score': dynamic_score,
                        'reasons': escalation_reasons
                    })
                    
                    # Update cluster parameters with escalation info
                    new_params = dict(cluster.cluster_parameters or {})
                    new_params.update({
                        'escalation_recommended': True,
                        'escalation_reasons': escalation_reasons,
                        'escalation_priority': 'URGENT'
                    })
                    cluster.cluster_parameters = new_params
                    db.add(cluster)
                    
                    # Update cluster summary with escalation info
                    summary = db.query(ClusterSummary).filter(ClusterSummary.cluster_id == cluster.id).first()
                    if summary:
                        escalation_text = f"""

🚨 ESCALATION RECOMMENDED:
• Reasons: {'; '.join(escalation_reasons)}
• Recommended Actions: Immediate team notification, priority triage, impact assessment
• Next Steps: Contact {COPILOT_TAXONOMY.get(taxonomy_category, {}).get('fix_team', 'Engineering Team')} within 24 hours
"""
                        current_summary = summary.summary_text or ''
                        summary.summary_text = current_summary + escalation_text
            
            # Log escalation summary
            if escalation_clusters:
                logger.info(f"🚨 {len(escalation_clusters)} clusters flagged for escalation")
                for cluster in escalation_clusters:
                    logger.info(f"   • Cluster {cluster['cluster_id']}: {cluster['size']} failures, score {cluster.get('dynamic_score', 'N/A')}")
            else:
                logger.info("✅ No critical escalations needed at this time")
            
            db.commit()
            db.close()
            
        except Exception as e:
            logger.error(f"Error generating escalation recommendations: {e}")

# Singleton instance
enhanced_cluster_manager = EnhancedClusterManager() 