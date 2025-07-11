from summarization import cluster_summarizer
from database import get_db_session
from models import Cluster

if __name__ == "__main__":
    db = get_db_session()
    try:
        clusters = db.query(Cluster).filter(Cluster.is_noise.is_(False)).all()
        print(f"Found {len(clusters)} clusters to summarize")
        success_count = 0
        for cluster in clusters:
            print(f"Generating summary for cluster {cluster.id}...")
            success = cluster_summarizer.summarize_cluster(cluster.id)
            if success:
                print(f"✅ Successfully regenerated summary for cluster {cluster.id}")
                success_count += 1
            else:
                print(f"❌ Failed to regenerate summary for cluster {cluster.id}")
        print(f"✅ Regenerated {success_count}/{len(clusters)} cluster summaries")
    finally:
        db.close()
