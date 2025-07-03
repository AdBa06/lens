#!/usr/bin/env python3
"""
Debug script to check cluster summaries
"""
from database import get_db_session
from models import ClusterSummary

def check_summaries():
    db = get_db_session()
    try:
        summaries = db.query(ClusterSummary).all()
        print(f"Found {len(summaries)} summaries")
        
        for s in summaries:
            print(f"Cluster {s.cluster_id}:")
            print(f"  Label: '{s.label}'")
            print(f"  Summary: '{s.summary_text[:100]}...'")
            print(f"  Root cause: '{s.root_cause[:100]}...'")
            print("---")
    finally:
        db.close()

if __name__ == "__main__":
    check_summaries()
