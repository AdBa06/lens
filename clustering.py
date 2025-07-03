import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime

# Try to import hdbscan, but don't fail if it's not available
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logging.warning("hdbscan not available, falling back to K-Means clustering")
from sqlalchemy.orm import Session
from models import Embedding, Cluster, ClusterAssignment
from database import get_db_session
from config import config

logger = logging.getLogger(__name__)

class ClusteringEngine:
    """Clustering engine for failure embeddings"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def perform_hdbscan_clustering(self, embeddings: np.ndarray, 
                                 min_cluster_size: int = None,
                                 min_samples: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform HDBSCAN clustering"""
        if not HDBSCAN_AVAILABLE:
            logger.error("HDBSCAN not available, cannot perform HDBSCAN clustering")
            return np.array([]), {}
            
        min_cluster_size = min_cluster_size or config.MIN_CLUSTER_SIZE
        min_samples = min_samples or config.MIN_SAMPLES
        
        try:
            # Normalize embeddings
            embeddings_scaled = self.scaler.fit_transform(embeddings)
            
            # Perform HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean'
            )
            
            cluster_labels = clusterer.fit_predict(embeddings_scaled)
            
            # Calculate clustering metrics
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            metrics = {
                'algorithm': 'hdbscan',
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'cluster_persistence': clusterer.cluster_persistence_.tolist() if hasattr(clusterer, 'cluster_persistence_') else None,
                'probabilities': clusterer.probabilities_.tolist() if hasattr(clusterer, 'probabilities_') else None
            }
            
            logger.info(f"HDBSCAN clustering completed: {n_clusters} clusters, {n_noise} noise points")
            return cluster_labels, metrics
            
        except Exception as e:
            logger.error(f"Error in HDBSCAN clustering: {e}")
            return np.array([]), {}
    
    def perform_dbscan_clustering(self, embeddings: np.ndarray, 
                                eps: float = None, 
                                min_samples: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform DBSCAN clustering"""
        eps = eps or 0.5  # Default eps value
        min_samples = min_samples or 5  # Default min_samples
        
        try:
            # Normalize embeddings
            embeddings_scaled = self.scaler.fit_transform(embeddings)
            
            # Perform DBSCAN clustering
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            cluster_labels = clusterer.fit_predict(embeddings_scaled)
            
            # Calculate clustering metrics
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            metrics = {
                'algorithm': 'dbscan',
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'eps': eps,
                'min_samples': min_samples
            }
            
            logger.info(f"DBSCAN clustering completed: {n_clusters} clusters, {n_noise} noise points")
            return cluster_labels, metrics
            
        except Exception as e:
            logger.error(f"Error in DBSCAN clustering: {e}")
            return np.array([]), {}

    def perform_kmeans_clustering(self, embeddings: np.ndarray, 
                                n_clusters: int = None, max_cluster_size: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform K-Means clustering with size constraints"""
        if n_clusters is None:
            # Use elbow method to determine optimal number of clusters
            n_clusters = self._find_optimal_k(embeddings)
        
        try:
            # Normalize embeddings
            embeddings_scaled = self.scaler.fit_transform(embeddings)
            
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(embeddings_scaled)
            
            # Apply size constraints if specified
            if max_cluster_size is not None:
                cluster_labels = self._enforce_cluster_size_limit(
                    embeddings_scaled, cluster_labels, max_cluster_size
                )
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(embeddings_scaled, cluster_labels)
            
            metrics = {
                'algorithm': 'kmeans_size_constrained' if max_cluster_size else 'kmeans',
                'n_clusters': len(set(cluster_labels)),
                'max_cluster_size': max_cluster_size,
                'silhouette_score': silhouette_avg,
                'inertia': kmeans.inertia_,
                'cluster_centers': kmeans.cluster_centers_.tolist()
            }
            
            logger.info(f"K-Means clustering completed: {len(set(cluster_labels))} clusters, max size: {max(np.bincount(cluster_labels))}")
            return cluster_labels, metrics
            
        except Exception as e:
            logger.error(f"Error in K-Means clustering: {e}")
            return np.array([]), {}
    
    def _find_optimal_k(self, embeddings: np.ndarray, max_k: int = 20) -> int:
        """Find optimal number of clusters using elbow method"""
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        inertias = []
        k_range = range(2, min(max_k + 1, len(embeddings) // 2))
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(embeddings_scaled)
                inertias.append(kmeans.inertia_)
            except Exception:
                continue
        
        if not inertias:
            return 5  # Default fallback
        
        # Find elbow using rate of change
        if len(inertias) >= 3:
            diffs = np.diff(inertias)
            diff_ratios = np.diff(diffs) / diffs[:-1]
            optimal_k_idx = np.argmax(diff_ratios) + 2  # +2 because we start from k=2
            optimal_k = k_range[optimal_k_idx]
        else:
            optimal_k = k_range[len(inertias) // 2]
        
        logger.info(f"Optimal K determined: {optimal_k}")
        return optimal_k
    
    def calculate_cluster_distances(self, embeddings: np.ndarray, 
                                  cluster_labels: np.ndarray,
                                  cluster_centers: np.ndarray = None) -> List[float]:
        """Calculate distances from points to cluster centroids"""
        distances = []
        
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise point
                distances.append(None)
                continue
            
            if cluster_centers is not None:
                # Use provided cluster centers (for K-Means)
                centroid = cluster_centers[label]
            else:
                # Calculate centroid for this cluster
                cluster_points = embeddings[cluster_labels == label]
                centroid = np.mean(cluster_points, axis=0)
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(embeddings[i] - centroid)
            distances.append(float(distance))
        
        return distances
    
    def _enforce_cluster_size_limit(self, embeddings: np.ndarray, 
                                  cluster_labels: np.ndarray, 
                                  max_size: int) -> np.ndarray:
        """Split clusters that exceed max_size"""
        new_labels = cluster_labels.copy()
        next_label = int(max(cluster_labels)) + 1
        
        # Check each cluster
        for label in set(cluster_labels):
            cluster_mask = cluster_labels == label
            cluster_size = int(np.sum(cluster_mask))
            
            if cluster_size > max_size:
                # Get embeddings for this cluster
                cluster_embeddings = embeddings[cluster_mask]
                cluster_indices = np.where(cluster_mask)[0]
                
                # Calculate how many sub-clusters we need
                n_subclusters = max(2, int(np.ceil(cluster_size / max_size)))
                
                # Ensure we don't try to create more clusters than we have points
                n_subclusters = min(n_subclusters, cluster_size)
                
                if n_subclusters > 1:
                    # Sub-cluster using K-means
                    sub_kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init='auto')
                    sub_labels = sub_kmeans.fit_predict(cluster_embeddings)
                    
                    # Assign new labels to the sub-clusters
                    for i, sub_label in enumerate(sub_labels):
                        new_labels[cluster_indices[i]] = next_label + sub_label
                    
                    next_label += n_subclusters
        
        return new_labels

class ClusterManager:
    """Manage clusters and assignments in the database"""
    
    def __init__(self):
        self.clustering_engine = ClusteringEngine()
    
    def perform_clustering(self, algorithm: str = "hdbscan", **kwargs) -> bool:
        """Perform clustering on all embeddings"""
        try:
            from embeddings import embedding_generator
            
            # Get embeddings for clustering
            embeddings_array, embedding_ids = embedding_generator.get_embeddings_for_clustering()
            
            if len(embeddings_array) == 0:
                logger.warning("No embeddings available for clustering")
                return False
            
            # Perform clustering
            if algorithm.lower() == "hdbscan":
                if HDBSCAN_AVAILABLE:
                    cluster_labels, metrics = self.clustering_engine.perform_hdbscan_clustering(
                        embeddings_array, **kwargs
                    )
                else:
                    logger.warning("HDBSCAN not available, falling back to DBSCAN")
                    cluster_labels, metrics = self.clustering_engine.perform_dbscan_clustering(
                        embeddings_array, **kwargs
                    )
            elif algorithm.lower() == "dbscan":
                cluster_labels, metrics = self.clustering_engine.perform_dbscan_clustering(
                    embeddings_array, **kwargs
                )
            elif algorithm.lower() == "kmeans":
                cluster_labels, metrics = self.clustering_engine.perform_kmeans_clustering(
                    embeddings_array, **kwargs
                )
            else:
                raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
            
            if len(cluster_labels) == 0:
                logger.error("Clustering failed")
                return False
            
            # Calculate distances to centroids
            cluster_centers = metrics.get('cluster_centers')
            if cluster_centers:
                cluster_centers = np.array(cluster_centers)
            
            distances = self.clustering_engine.calculate_cluster_distances(
                embeddings_array, cluster_labels, cluster_centers
            )
            
            # Store clustering results
            return self._store_clustering_results(
                embedding_ids, cluster_labels, distances, metrics
            )
            
        except Exception as e:
            logger.error(f"Error in clustering process: {e}")
            return False
    
    def _store_clustering_results(self, embedding_ids: List[int], 
                                cluster_labels: np.ndarray,
                                distances: List[float],
                                metrics: Dict[str, Any]) -> bool:
        """Store clustering results in database"""
        db = get_db_session()
        
        try:
            # Clear existing clusters and assignments
            db.query(ClusterAssignment).delete()
            db.query(Cluster).delete()
            db.commit()
            
            # Create cluster records
            unique_labels = set(cluster_labels)
            cluster_map = {}
            
            for label in unique_labels:
                cluster_size = list(cluster_labels).count(label)
                is_noise = (label == -1)
                
                # Convert noise cluster to "Uncategorized" cluster - treat as regular cluster
                cluster = Cluster(
                    cluster_label=int(label),
                    cluster_algorithm=metrics['algorithm'],
                    cluster_parameters=metrics,
                    size=cluster_size,
                    is_noise=False  # Mark all clusters as regular clusters, including former "noise"
                )
                
                db.add(cluster)
                db.flush()  # Get the ID
                cluster_map[label] = cluster.id
            
            # Create cluster assignments
            for i, (embedding_id, label, distance) in enumerate(zip(embedding_ids, cluster_labels, distances)):
                assignment = ClusterAssignment(
                    embedding_id=embedding_id,
                    cluster_id=cluster_map[label],
                    distance_to_centroid=distance,
                    membership_probability=metrics.get('probabilities', [None] * len(embedding_ids))[i]
                )
                db.add(assignment)
            
            db.commit()
            logger.info(f"Successfully stored clustering results: {len(unique_labels)} clusters")
            return True
            
        except Exception as e:
            logger.error(f"Error storing clustering results: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get clustering statistics"""
        db = get_db_session()
        
        try:
            clusters = db.query(Cluster).all()
            
            if not clusters:
                return {}
            
            total_clusters = len([c for c in clusters if not c.is_noise])
            noise_clusters = len([c for c in clusters if c.is_noise])
            total_points = sum(c.size for c in clusters)
            noise_points = sum(c.size for c in clusters if c.is_noise)
            
            cluster_sizes = [c.size for c in clusters if not c.is_noise]
            
            stats = {
                'total_clusters': total_clusters,
                'noise_clusters': noise_clusters,
                'total_points': total_points,
                'noise_points': noise_points,
                'average_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
                'median_cluster_size': np.median(cluster_sizes) if cluster_sizes else 0,
                'largest_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
                'smallest_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
                'algorithm': clusters[0].cluster_algorithm if clusters else None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cluster statistics: {e}")
            return {}
        finally:
            db.close()

# Singleton instance
cluster_manager = ClusterManager() 