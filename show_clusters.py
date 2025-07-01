#!/usr/bin/env python3
"""
Simple script to show cluster analysis results
"""

from database import SessionLocal
from models import Cluster, ClusterAssignment, Embedding, TelemetryEvent, FailureFingerprint
import json

def show_clusters():
    print("🎯 CLUSTER ANALYSIS RESULTS")
    print("=" * 80)
    
    db = SessionLocal()
    
    # Get all clusters
    clusters = db.query(Cluster).all()
    print(f"Found {len(clusters)} clusters from machine learning analysis:\n")
    
    for cluster in clusters:
        print(f"📊 CLUSTER {cluster.cluster_label}")
        print(f"   Size: {cluster.size} similar failures")
        print(f"   Algorithm: {cluster.cluster_algorithm}")
        print()
        
        # Get all events in this cluster
        assignments = db.query(ClusterAssignment).filter(ClusterAssignment.cluster_id == cluster.id).all()
        
        print(f"   Events in this cluster:")
        for i, assignment in enumerate(assignments, 1):
            embedding = db.query(Embedding).filter(Embedding.id == assignment.embedding_id).first()
            if embedding:
                event = db.query(TelemetryEvent).filter(TelemetryEvent.id == embedding.event_id).first()
                if event:
                    # Get error details
                    fingerprint = db.query(FailureFingerprint).filter(FailureFingerprint.event_id == event.id).first()
                    
                    print(f"     {i}. {event.evaluation_id}")
                    print(f"        Prompt: {event.customer_prompt}")
                    if fingerprint:
                        print(f"        Plugin: {fingerprint.plugin_name}")
                        print(f"        Error: {fingerprint.status_code} - {fingerprint.error_message}")
                        print(f"        Endpoint: {fingerprint.endpoint}")
                    print()
        
        print("-" * 80)
        print()
    
    db.close()

if __name__ == "__main__":
    show_clusters() 