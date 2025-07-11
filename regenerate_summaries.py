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


import subprocess
import sys

def regenerate_summaries():
    """Regenerate all cluster summaries by spawning a fresh process for each cluster (like the working single-cluster script)"""
    db = get_db_session()
    try:
        clusters = db.query(Cluster).filter(Cluster.is_noise.is_(False)).all()
        print(f"Found {len(clusters)} clusters to summarize")
        success_count = 0
        for cluster in clusters:
            print(f"Generating summary for cluster {cluster.id}...")
            # Call the single-cluster script as a subprocess
            result = subprocess.run([sys.executable, "regenerate_cluster1.py", str(cluster.id)], capture_output=True, text=True)
            print(result.stdout)
            if result.returncode == 0 and "Successfully" in result.stdout:
                success_count += 1
            else:
                print(f"❌ Failed to summarize cluster {cluster.id}")
        print(f"✅ Regenerated {success_count}/{len(clusters)} cluster summaries")
    finally:
        db.close()

if __name__ == "__main__":
    regenerate_summaries()
