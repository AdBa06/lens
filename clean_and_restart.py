#!/usr/bin/env python3
"""Clean database and restart with only synthetic data"""

from database import get_db_session
from models import (
    TelemetryEvent, Embedding, Cluster, ClusterAssignment, 
    ClusterSummary, FailureFingerprint
)

def main():
    print("🧹 Cleaning database to keep only synthetic data...")
    
    db = get_db_session()
    try:
        # Check current state
        total_events = db.query(TelemetryEvent).count()
        synthetic_events = db.query(TelemetryEvent).filter(
            TelemetryEvent.evaluation_id.like('synthetic_row_%')
        ).count()
        
        print(f"📊 Current state:")
        print(f"   Total events: {total_events}")
        print(f"   Synthetic events: {synthetic_events}")
        print(f"   Old events to remove: {total_events - synthetic_events}")
        
        if synthetic_events == 0:
            print("❌ No synthetic events found! Something went wrong.")
            return False
        
        # Clear all analysis data
        print("🗑️  Clearing all embeddings, clusters, and summaries...")
        db.query(ClusterAssignment).delete()
        db.query(ClusterSummary).delete()
        db.query(Cluster).delete()
        db.query(Embedding).delete()
        db.query(FailureFingerprint).delete()
        
        # Delete non-synthetic events
        if total_events > synthetic_events:
            print("🗑️  Removing old non-synthetic events...")
            deleted = db.query(TelemetryEvent).filter(
                ~TelemetryEvent.evaluation_id.like('synthetic_row_%')
            ).delete(synchronize_session=False)
            print(f"   Deleted {deleted} old events")
        
        db.commit()
        
        # Verify final state
        final_count = db.query(TelemetryEvent).count()
        print(f"✅ Database cleaned! Final event count: {final_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error cleaning database: {e}")
        db.rollback()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Next steps:")
        print("1. Run embeddings: python run_embeddings.py") 
        print("2. Run clustering: python -c \"from clustering import cluster_manager; cluster_manager.perform_clustering('hdbscan')\"")
        print("3. Generate summaries: python generate_summaries.py")
        print("4. View dashboard: http://localhost:8080") 