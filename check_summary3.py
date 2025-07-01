from database import SessionLocal
from models import ClusterSummary

db = SessionLocal()
try:
    s3 = db.query(ClusterSummary).filter(ClusterSummary.cluster_id == 3).first()
    if s3:
        print(f"Cluster 3 summary exists: True")
        print(f"Summary text length: {len(s3.summary_text) if s3.summary_text else 0}")
        print(f"Summary text: '{s3.summary_text}'")
        print(f"Root cause: '{s3.root_cause}'")
    else:
        print("No summary found for cluster 3")
finally:
    db.close() 