from database import get_db_session
from models import TelemetryEvent, Embedding, Cluster, ClusterSummary

db = get_db_session()
try:
    events = db.query(TelemetryEvent).count()
    embeddings = db.query(Embedding).count()
    clusters = db.query(Cluster).count()
    summaries = db.query(ClusterSummary).count()
    
    print(f"📊 Database Status:")
    print(f"   Events: {events}")
    print(f"   Embeddings: {embeddings}")
    print(f"   Clusters: {clusters}")
    print(f"   Summaries: {summaries}")
    
    if clusters > 0:
        print(f"\n🔍 Cluster Details:")
        cluster_list = db.query(Cluster).all()
        for cluster in cluster_list[:5]:  # Show first 5
            print(f"   Cluster {cluster.id}: {cluster.size} events, algorithm: {cluster.cluster_algorithm}")
            if cluster.summary and cluster.summary.summary_text:
                print(f"      Summary: {cluster.summary.summary_text[:100]}...")
            else:
                print(f"      Summary: Not generated yet")
    else:
        print(f"\n⚠️  No clusters found in database!")
        
finally:
    db.close() 