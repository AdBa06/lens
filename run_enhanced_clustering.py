#!/usr/bin/env python3
"""
Run Enhanced Clustering Pipeline
Test the new hybrid approach with domain-guided clustering and LLM validation
"""

import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_progress(message):
    """Print progress with timestamp and emoji"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    logger.info(message)

def compare_clustering_results():
    """Compare original vs enhanced clustering results"""
    from database import get_db_session
    from models import Cluster, ClusterSummary
    
    db = get_db_session()
    try:
        clusters = db.query(Cluster).all()
        
        if not clusters:
            print_progress("❌ No clusters found in database")
            return
        
        print("\n" + "="*80)
        print("🔍 CLUSTERING COMPARISON RESULTS")
        print("="*80)
        
        # Group by algorithm
        algorithms = {}
        for cluster in clusters:
            algo = cluster.cluster_algorithm
            if algo not in algorithms:
                algorithms[algo] = []
            algorithms[algo].append(cluster)
        
        for algo, cluster_list in algorithms.items():
            print(f"\n📊 {algo.upper()} RESULTS:")
            print(f"   Total Clusters: {len([c for c in cluster_list if not c.is_noise])}")
            print(f"   Noise Points: {sum(c.size for c in cluster_list if c.is_noise)}")
            print(f"   Total Points: {sum(c.size for c in cluster_list)}")
            
            # Show top clusters
            sorted_clusters = sorted([c for c in cluster_list if not c.is_noise], 
                                   key=lambda x: x.size, reverse=True)[:5]
            
            for i, cluster in enumerate(sorted_clusters, 1):
                summary = db.query(ClusterSummary).filter(
                    ClusterSummary.cluster_id == cluster.id
                ).first()
                
                taxonomy_category = cluster.cluster_parameters.get('taxonomy_category', 'Unknown')
                
                print(f"\n   🎯 Cluster {i} (Size: {cluster.size})")
                if taxonomy_category != 'Unknown':
                    print(f"      📋 Category: {taxonomy_category}")
                if summary and summary.summary_text:
                    print(f"      📝 Summary: {summary.summary_text[:150]}...")
                if summary and summary.root_cause:
                    print(f"      🔍 Root Cause: {summary.root_cause[:150]}...")
        
    finally:
        db.close()

def main():
    """Main execution function"""
    print_progress("🚀 Starting Enhanced Clustering Pipeline")
    
    try:
        # Import enhanced clustering
        from enhanced_clustering import enhanced_cluster_manager
        
        print_progress("🔄 Running enhanced clustering algorithm...")
        
        # Perform enhanced clustering
        success = enhanced_cluster_manager.perform_enhanced_clustering()
        
        if success:
            print_progress("✅ Enhanced clustering completed successfully!")
            
            # Show comparison results
            compare_clustering_results()
            
            print("\n" + "="*80)
            print("🎉 ENHANCED CLUSTERING PIPELINE COMPLETED!")
            print("="*80)
            print("📊 Check the web dashboard at: http://localhost:8080")
            print("🔍 Compare with previous results to see improvements")
            print("💡 Enhanced features:")
            print("   • Domain-guided initial clustering")
            print("   • LLM validation of cluster quality")
            print("   • Business context and team assignments")
            print("   • Improved root cause analysis")
            
        else:
            print_progress("❌ Enhanced clustering failed")
            return False
            
    except Exception as e:
        logger.error(f"Error in enhanced clustering pipeline: {e}")
        print_progress(f"❌ Pipeline failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 