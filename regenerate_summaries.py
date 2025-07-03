#!/usr/bin/env python3
"""
Regenerate cluster summaries with labels and recommendations
"""
import logging
from summarization import cluster_summarizer
from database import get_db_session
from models import Cluster

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def regenerate_summaries():
    """Regenerate all cluster summaries with new label and recommendations fields"""
    db = get_db_session()
    try:
        clusters = db.query(Cluster).filter(Cluster.is_noise.is_(False)).all()
        print(f"Found {len(clusters)} clusters to summarize")
        
        success_count = 0
        for cluster in clusters:
            print(f"Generating summary for cluster {cluster.id}...")
            try:
                if cluster_summarizer.summarize_cluster(cluster.id):
                    success_count += 1
                    print(f"✅ Successfully summarized cluster {cluster.id}")
                else:
                    print(f"❌ Failed to summarize cluster {cluster.id}")
            except Exception as e:
                print(f"❌ Error summarizing cluster {cluster.id}: {e}")
        
        print(f"✅ Regenerated {success_count}/{len(clusters)} cluster summaries")
        
    finally:
        db.close()

if __name__ == "__main__":
    regenerate_summaries()
