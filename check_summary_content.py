#!/usr/bin/env python3
"""
Debug script to check cluster summary content
"""
from database import get_db_session
from models import ClusterSummary

def check_summary_content():
    db = get_db_session()
    try:
        s = db.query(ClusterSummary).first()
        if s:
            print("First summary:")
            print(f'  Cluster ID: {s.cluster_id}')
            print(f'  Label: "{s.label}"')
            print(f'  Summary: "{s.summary_text}"')
            print(f'  Root cause: "{s.root_cause}"')
            print(f'  Recommendations: {s.recommendations}')
        else:
            print("No summaries found")
    finally:
        db.close()

if __name__ == "__main__":
    check_summary_content()
