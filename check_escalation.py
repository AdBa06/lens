#!/usr/bin/env python3

import json
from database import get_db_session
from models import Cluster, ClusterSummary

def check_escalation_status():
    """Check escalation status of all clusters"""
    db = get_db_session()
    try:
        clusters = db.query(Cluster).filter(Cluster.cluster_algorithm == 'enhanced_hybrid').all()
        
        print("🔍 Cluster Escalation Status:")
        print("=" * 50)
        
        for cluster in clusters:
            print(f"\n📊 Cluster {cluster.id}:")
            print(f"   Size: {cluster.size} events")
            print(f"   Algorithm: {cluster.cluster_algorithm}")
            
            # Check cluster parameters
            if cluster.cluster_parameters:
                params = cluster.cluster_parameters
                print(f"   Taxonomy: {params.get('taxonomy_category', 'None')}")
                print(f"   Priority Score: {params.get('priority_score', 'None')}")
                print(f"   Escalation: {params.get('escalation_recommended', 'None')}")
                print(f"   Business Impact: {params.get('business_impact', 'None')}")
            else:
                print("   ⚠️  No cluster parameters found")
            
            # Check summary
            summary = db.query(ClusterSummary).filter(ClusterSummary.cluster_id == cluster.id).first()
            if summary:
                print(f"   Summary: {summary.summary_text[:100]}...")
            else:
                print("   ⚠️  No summary found")
                
    finally:
        db.close()

if __name__ == "__main__":
    check_escalation_status() 