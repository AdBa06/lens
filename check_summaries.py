#!/usr/bin/env python3

from database import get_db_session
from models import Cluster, ClusterSummary

def check_summaries():
    """Check which clusters have AI summaries"""
    db = get_db_session()
    try:
        clusters = db.query(Cluster).filter(Cluster.cluster_algorithm == 'enhanced_hybrid').all()
        
        print("🔍 Cluster Summary Status:")
        print("=" * 50)
        
        for cluster in clusters:
            print(f"\n📊 Cluster {cluster.id} (Size: {cluster.size}):")
            
            # Check for summary
            summary = db.query(ClusterSummary).filter(ClusterSummary.cluster_id == cluster.id).first()
            if summary:
                print(f"   ✅ Has AI Summary: {len(summary.summary_text)} chars")
                print(f"   📝 Preview: {summary.summary_text[:100]}...")
                if summary.root_cause:
                    print(f"   🔍 Has Root Cause: {len(summary.root_cause)} chars")
            else:
                print(f"   ❌ NO AI Summary")
                
    finally:
        db.close()

if __name__ == "__main__":
    check_summaries() 