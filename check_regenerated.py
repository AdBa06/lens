#!/usr/bin/env python3
"""
Check regenerated summaries
"""
from database import get_db_session
from models import ClusterSummary

def check_regenerated():
    db = get_db_session()
    try:
        summaries = db.query(ClusterSummary).filter(ClusterSummary.cluster_id.in_([1,2,3,4,5])).all()
        for s in summaries:
            print(f"Cluster {s.cluster_id}:")
            print(f"  Label: '{s.label}'")
            print(f"  Summary: '{s.summary_text[:150]}...'")
            print("---")
    finally:
        db.close()

if __name__ == "__main__":
    check_regenerated()
