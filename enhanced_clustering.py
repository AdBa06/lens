import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from datetime import datetime
import json

# Try to import hdbscan, but don't fail if it's not available
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logging.warning("hdbscan not available, falling back to K-Means clustering")

from sqlalchemy.orm import Session
from models import Embedding, Cluster, ClusterAssignment, TelemetryEvent
from database import get_db_session
from config import config

logger = logging.getLogger(__name__)

class EnhancedClusteringEngine:
    def _split_large_clusters(self, vectors, labels, max_cluster_size, clustering_method, sample_weights=None, depth=0):
        """Recursively split clusters larger than max_cluster_size using the same clustering method."""
        import numpy as np
        new_labels = labels.copy()
        next_label = new_labels.max() + 1 if len(new_labels) > 0 else 0
        unique_labels = set(new_labels)
        for label in unique_labels:
            if label == -1:
                continue  # skip noise
            indices = np.where(new_labels == label)[0]
            if len(indices) > max_cluster_size:
                # Re-cluster this cluster's members
                sub_vectors = vectors[indices]
                sub_weights = sample_weights[indices] if sample_weights is not None else None
                # Use the same clustering method as before
                if clustering_method == 'hdbscan' and HDBSCAN_AVAILABLE:
                    result = self._try_hdbscan_clustering(sub_vectors, min_cluster_size=max(2, max_cluster_size//2), sample_weights=sub_weights)
                elif clustering_method == 'kmeans':
                    result = self._try_kmeans_clustering(sub_vectors, sample_weights=sub_weights)
                elif clustering_method == 'dbscan':
                    result = self._try_dbscan_clustering(sub_vectors, sample_weights=sub_weights)
                else:
                    continue
                sub_labels = result['labels']
                # Remap sub-labels to global label space
                sub_unique = set(sub_labels)
                sub_map = {subl: (label if i == 0 else next_label + i - 1) for i, subl in enumerate(sub_unique)}
                for i, idx in enumerate(indices):
                    if sub_labels[i] == -1:
                        new_labels[idx] = -1
                    else:
                        new_labels[idx] = sub_map[sub_labels[i]]
                next_label += len(sub_unique) - 1
        # Check if any clusters are still too large
        if any((new_labels == l).sum() > max_cluster_size for l in set(new_labels) if l != -1):
            # Recurse
            return self._split_large_clusters(vectors, new_labels, max_cluster_size, clustering_method, sample_weights, depth+1)
        return new_labels
    """Enhanced clustering engine with business intelligence"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.business_weight_factors = {
            "authentication": 2.0,  # Weight auth failures higher
            "critical_services": 1.8,  # Critical service failures
            "integration": 1.5,  # API/integration issues
            "permissions": 1.3,  # Permission-related
            "general": 1.0  # Default weight
        }
    
    def calculate_business_weights(self, events: List[TelemetryEvent]) -> np.ndarray:
        """Calculate business importance weights for events"""
        weights = np.ones(len(events))
        
        for i, event in enumerate(events):
            text = f"{event.customer_prompt or ''} {json.dumps(event.skill_output) if event.skill_output else ''}".lower()
            
            # Apply business weights
            if any(word in text for word in ["signin", "login", "auth", "forbidden", "unauthorized"]):
                weights[i] = self.business_weight_factors["authentication"]
            elif any(word in text for word in ["500", "503", "502", "timeout", "unavailable"]):
                weights[i] = self.business_weight_factors["critical_services"]
            elif any(word in text for word in ["api", "plugin", "integration", "connector"]):
                weights[i] = self.business_weight_factors["integration"]
            elif any(word in text for word in ["permission", "access", "scope", "consent"]):
                weights[i] = self.business_weight_factors["permissions"]
        
        return weights
    
    def perform_enhanced_clustering(self, min_cluster_size: int = None,
                                  min_samples: int = None,
                                  use_business_weights: bool = True) -> Dict[str, Any]:
        """Perform enhanced clustering with business intelligence"""
        
        db = get_db_session()
        try:
            # Get embeddings with their events
            embeddings_query = db.query(Embedding).join(TelemetryEvent).all()
            
            if len(embeddings_query) < 5:
                logger.warning("Not enough embeddings for clustering")
                return {"error": "Insufficient data for clustering"}
            
            # Extract vectors and events
            vectors = np.array([emb.embedding_vector for emb in embeddings_query])
            events = [emb.event for emb in embeddings_query]
            embedding_ids = [emb.id for emb in embeddings_query]
            
            # Apply business weights if requested
            sample_weights = None
            if use_business_weights:
                sample_weights = self.calculate_business_weights(events)
                logger.info(f"Applied business weights, avg weight: {sample_weights.mean():.2f}")
            
            # Try multiple clustering approaches and pick the best
            results = []
            
            # 1. HDBSCAN (if available)
            if HDBSCAN_AVAILABLE:
                hdbscan_result = self._try_hdbscan_clustering(
                    vectors, min_cluster_size, min_samples, sample_weights
                )
                if hdbscan_result["success"]:
                    results.append(("hdbscan", hdbscan_result))
            
            # 2. K-Means with optimal K
            kmeans_result = self._try_kmeans_clustering(vectors, sample_weights)
            if kmeans_result["success"]:
                results.append(("kmeans", kmeans_result))
            
            # 3. DBSCAN
            dbscan_result = self._try_dbscan_clustering(vectors, sample_weights)
            if dbscan_result["success"]:
                results.append(("dbscan", dbscan_result))
            
            if not results:
                return {"error": "All clustering methods failed"}
            
            # Select best clustering result
            best_method, best_result = self._select_best_clustering(results, vectors)
            logger.info(f"Selected {best_method} clustering with silhouette score: {best_result['silhouette_score']:.3f}")
            
            # Enforce max cluster size (10% of dataset)
            total_events = len(vectors)
            max_cluster_size = max(1, int(total_events * 0.10))
            final_labels = self._split_large_clusters(
                vectors, np.array(best_result["labels"]), max_cluster_size, best_method, sample_weights
            )
            # Save clustering results
            cluster_mapping = self._save_clustering_results(
                db, embedding_ids, final_labels, best_method, best_result
            )
            return {
                "success": True,
                "method": best_method,
                "num_clusters": len(set(final_labels)) - (1 if -1 in final_labels else 0),
                "silhouette_score": best_result["silhouette_score"],
                "cluster_mapping": cluster_mapping,
                "business_weighted": use_business_weights,
                "max_cluster_size": max_cluster_size
            }
            
        except Exception as e:
            logger.error(f"Enhanced clustering failed: {e}")
            return {"error": str(e)}
        finally:
            db.close()
    
    def _try_hdbscan_clustering(self, vectors: np.ndarray, min_cluster_size: int = None,
                               min_samples: int = None, sample_weights: np.ndarray = None) -> Dict[str, Any]:
        """Try HDBSCAN clustering"""
        try:
            min_cluster_size = min_cluster_size or max(5, len(vectors) // 20)
            min_samples = min_samples or max(3, min_cluster_size // 2)
            
            vectors_scaled = self.scaler.fit_transform(vectors)
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            
            labels = clusterer.fit_predict(vectors_scaled, sample_weight=sample_weights)
            
            # Calculate metrics
            if len(set(labels)) > 1:
                silhouette = silhouette_score(vectors_scaled, labels)
                calinski_harabasz = calinski_harabasz_score(vectors_scaled, labels)
            else:
                silhouette = -1
                calinski_harabasz = 0
            
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_points = sum(1 for label in labels if label == -1)
            
            return {
                "success": True,
                "labels": labels,
                "num_clusters": num_clusters,
                "noise_points": noise_points,
                "silhouette_score": silhouette,
                "calinski_harabasz_score": calinski_harabasz,
                "parameters": {
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples
                }
            }
            
        except Exception as e:
            logger.error(f"HDBSCAN clustering failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _try_kmeans_clustering(self, vectors: np.ndarray, sample_weights: np.ndarray = None) -> Dict[str, Any]:
        """Try K-Means clustering with optimal K"""
        try:
            vectors_scaled = self.scaler.fit_transform(vectors)
            
            # Find optimal K using silhouette analysis
            best_k = 2
            best_silhouette = -1
            best_labels = None
            
            max_k = min(10, len(vectors) // 3)
            
            for k in range(2, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(vectors_scaled, sample_weight=sample_weights)
                
                silhouette = silhouette_score(vectors_scaled, labels)
                
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_k = k
                    best_labels = labels
            
            calinski_harabasz = calinski_harabasz_score(vectors_scaled, best_labels)
            
            return {
                "success": True,
                "labels": best_labels,
                "num_clusters": best_k,
                "noise_points": 0,
                "silhouette_score": best_silhouette,
                "calinski_harabasz_score": calinski_harabasz,
                "parameters": {"n_clusters": best_k}
            }
            
        except Exception as e:
            logger.error(f"K-Means clustering failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _try_dbscan_clustering(self, vectors: np.ndarray, sample_weights: np.ndarray = None) -> Dict[str, Any]:
        """Try DBSCAN clustering"""
        try:
            vectors_scaled = self.scaler.fit_transform(vectors)
            
            # Find optimal eps using k-distance graph heuristic
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=5)
            neighbors_fit = neighbors.fit(vectors_scaled)
            distances, indices = neighbors_fit.kneighbors(vectors_scaled)
            distances = np.sort(distances[:, 4], axis=0)
            
            # Use elbow method to find eps
            eps = distances[int(len(distances) * 0.8)]
            
            dbscan = DBSCAN(eps=eps, min_samples=5)
            labels = dbscan.fit_predict(vectors_scaled, sample_weight=sample_weights)
            
            # Calculate metrics
            if len(set(labels)) > 1:
                silhouette = silhouette_score(vectors_scaled, labels)
                calinski_harabasz = calinski_harabasz_score(vectors_scaled, labels)
            else:
                silhouette = -1
                calinski_harabasz = 0
            
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_points = sum(1 for label in labels if label == -1)
            
            return {
                "success": True,
                "labels": labels,
                "num_clusters": num_clusters,
                "noise_points": noise_points,
                "silhouette_score": silhouette,
                "calinski_harabasz_score": calinski_harabasz,
                "parameters": {"eps": eps, "min_samples": 5}
            }
            
        except Exception as e:
            logger.error(f"DBSCAN clustering failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _select_best_clustering(self, results: List[Tuple[str, Dict]], vectors: np.ndarray) -> Tuple[str, Dict]:
        """Select the best clustering result based on metrics"""
        if len(results) == 1:
            return results[0]
        
        best_score = -1
        best_result = None
        
        for method, result in results:
            # Composite score: silhouette + cluster validity
            score = result["silhouette_score"]
            
            # Penalize too many or too few clusters
            num_clusters = result["num_clusters"]
            optimal_clusters = max(2, min(10, len(vectors) // 15))
            cluster_penalty = abs(num_clusters - optimal_clusters) / optimal_clusters
            score -= cluster_penalty * 0.2
            
            # Bonus for HDBSCAN (better for irregular clusters)
            if method == "hdbscan":
                score += 0.1
            
            if score > best_score:
                best_score = score
                best_result = (method, result)
        
        return best_result
    
    def _save_clustering_results(self, db: Session, embedding_ids: List[int], 
                               labels: np.ndarray, method: str, result: Dict) -> Dict[str, int]:
        """Save clustering results to database"""
        try:
            # Clear existing clusters and assignments
            db.query(ClusterAssignment).delete()
            db.query(Cluster).delete()
            db.commit()
            
            # Create clusters
            cluster_mapping = {}
            unique_labels = set(labels)
            
            for label in unique_labels:
                cluster_size = sum(1 for l in labels if l == label)
                is_noise = (label == -1)
                
                cluster = Cluster(
                    cluster_label=int(label),
                    cluster_algorithm=method,
                    cluster_parameters=result["parameters"],
                    size=cluster_size,
                    is_noise=is_noise
                )
                db.add(cluster)
                db.flush()
                cluster_mapping[int(label)] = cluster.id
            
            # Create assignments
            for embedding_id, label in zip(embedding_ids, labels):
                if int(label) in cluster_mapping:
                    assignment = ClusterAssignment(
                        embedding_id=embedding_id,
                        cluster_id=cluster_mapping[int(label)]
                    )
                    db.add(assignment)
            
            db.commit()
            logger.info(f"Saved {len(unique_labels)} clusters with {len(embedding_ids)} assignments")
            
            return cluster_mapping
            
        except Exception as e:
            logger.error(f"Error saving clustering results: {e}")
            db.rollback()
            raise

# Enhanced singleton instance
enhanced_cluster_manager = EnhancedClusteringEngine()
