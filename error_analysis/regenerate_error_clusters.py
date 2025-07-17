
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import get_db_session
from models import Cluster
from summarization import cluster_summarizer

if __name__ == "__main__":
    db = get_db_session()
    try:
        clusters = db.query(Cluster).filter(Cluster.is_noise.is_(False)).all()
        print(f"Found {len(clusters)} error clusters to analyze")
        success_count = 0
        for cluster in clusters:
            print(f"Analyzing error cluster {cluster.id}...")
            result = cluster_summarizer.summarize_cluster(cluster.id)
            if result:
                print(f"✅ Root cause and fixes generated for cluster {cluster.id}")
                success_count += 1
            else:
                print(f"❌ Failed to analyze cluster {cluster.id}")
        print(f"✅ Analyzed {success_count}/{len(clusters)} error clusters")
    finally:
        db.close()
