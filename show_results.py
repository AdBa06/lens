from database import get_db_session
from models import Cluster, ClusterSummary
from sqlalchemy.orm import joinedload

db = get_db_session()
try:
    # Get all clusters with their summaries
    clusters = db.query(Cluster).options(joinedload(Cluster.summary)).order_by(Cluster.size.desc()).all()
    
    print("🤖 COPILOT FAILURE ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total Clusters Found: {len(clusters)}")
    print("=" * 60)
    
    for i, cluster in enumerate(clusters, 1):
        print(f"\n📊 CLUSTER {i}")
        print(f"   Size: {cluster.size} failures")
        print(f"   Algorithm: {cluster.cluster_algorithm}")
        
        if cluster.summary:
            summary = cluster.summary
            if summary.summary_text:
                print(f"   📝 Summary: {summary.summary_text}")
            if summary.root_cause:
                print(f"   🔍 Root Cause: {summary.root_cause}")
            if summary.common_plugins:
                plugins = ', '.join(summary.common_plugins[:3])
                print(f"   🔧 Common Plugins: {plugins}")
            if summary.common_endpoints:
                endpoints = ', '.join(summary.common_endpoints[:3])
                print(f"   🌐 Common Endpoints: {endpoints}")
            if summary.sample_prompts:
                prompt = summary.sample_prompts[0][:100] + "..." if len(summary.sample_prompts[0]) > 100 else summary.sample_prompts[0]
                print(f"   💬 Sample Prompt: {prompt}")
        else:
            print("   ❌ No summary generated")
        
        print("-" * 60)

finally:
    db.close()

print("\n🎉 Analysis Complete!") 