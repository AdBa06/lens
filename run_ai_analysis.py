#!/usr/bin/env python3
"""
Simplified AI Analysis for Copilot Failures
Uses KMeans clustering (no C++ compilation needed)
"""

import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are available"""
    missing = []
    
    try:
        import openai
    except ImportError:
        missing.append("openai")
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing.append("sentence-transformers")
    
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        missing.append("scikit-learn")
    
    return missing

def generate_embeddings():
    """Generate embeddings for failure events"""
    try:
        from database import SessionLocal
        from models import TelemetryEvent, Embedding
        from sentence_transformers import SentenceTransformer
        
        db = SessionLocal()
        
        # Get events without embeddings
        events = db.query(TelemetryEvent).all()
        logger.info(f"Found {len(events)} events to process")
        
        if not events:
            logger.warning("No events found. Try running demo.py first")
            return False
        
        # Load sentence transformer model
        logger.info("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings
        texts = []
        for event in events:
            # Combine customer prompt and skill output for embedding
            skill_output_str = json.dumps(event.skill_output) if isinstance(event.skill_output, dict) else str(event.skill_output)
            combined_text = f"{event.customer_prompt} {skill_output_str}"
            texts.append(combined_text)
        
        logger.info("Generating embeddings...")
        embeddings = model.encode(texts)
        
        # Save embeddings to database
        for i, event in enumerate(events):
            # Check if embedding already exists
            existing = db.query(Embedding).filter(Embedding.event_id == event.id).first()
            if not existing:
                embedding = Embedding(
                    event_id=event.id,
                    embedding_vector=embeddings[i].tolist(),
                    embedding_model='all-MiniLM-L6-v2',
                    embedding_source='customer_prompt+skill_output',
                    created_at=datetime.utcnow()
                )
                db.add(embedding)
        
        db.commit()
        db.close()
        
        logger.info(f"✅ Generated embeddings for {len(events)} events")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error generating embeddings: {e}")
        return False

def cluster_embeddings():
    """Cluster embeddings using KMeans"""
    try:
        from database import SessionLocal
        from models import Embedding, Cluster, ClusterAssignment
        from sklearn.cluster import KMeans
        import numpy as np
        
        db = SessionLocal()
        
        # Get all embeddings
        embeddings = db.query(Embedding).all()
        logger.info(f"Found {len(embeddings)} embeddings to cluster")
        
        if len(embeddings) < 5:
            logger.warning("Need at least 5 embeddings for clustering")
            return False
        
        # Prepare data for clustering
        vectors = np.array([emb.embedding_vector for emb in embeddings])
        
        # Determine optimal number of clusters (simple heuristic)
        n_clusters = min(max(2, len(embeddings) // 5), 10)
        logger.info(f"Using {n_clusters} clusters")
        
        # Perform KMeans clustering
        logger.info("Performing KMeans clustering...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(vectors)
        
        # Save clusters to database
        for cluster_id in range(n_clusters):
            cluster_embeddings = [emb for i, emb in enumerate(embeddings) if cluster_labels[i] == cluster_id]
            
            if cluster_embeddings:
                # Create cluster
                cluster = Cluster(
                    cluster_label=cluster_id,
                    size=len(cluster_embeddings),
                    cluster_algorithm='kmeans',
                    cluster_parameters={'n_clusters': n_clusters, 'centroid': kmeans.cluster_centers_[cluster_id].tolist()},
                    created_at=datetime.utcnow()
                )
                db.add(cluster)
                db.commit()
                
                # Assign embeddings to cluster
                for emb in cluster_embeddings:
                    assignment = ClusterAssignment(
                        embedding_id=emb.id,
                        cluster_id=cluster.id,
                        distance_to_centroid=float(np.linalg.norm(np.array(emb.embedding_vector) - kmeans.cluster_centers_[cluster_id]))
                    )
                    db.add(assignment)
        
        db.commit()
        db.close()
        
        logger.info(f"✅ Created {n_clusters} clusters")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error clustering embeddings: {e}")
        return False

def generate_summaries():
    """Generate AI summaries for clusters"""
    try:
        from database import SessionLocal
        from models import Cluster, ClusterSummary, ClusterAssignment, Embedding, TelemetryEvent
        from config import config
        import openai
        
        if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "sk-your-actual-openai-key-here":
            logger.warning("⚠️  No OpenAI API key configured. Skipping AI summaries.")
            return False
        
        # Set up OpenAI
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        db = SessionLocal()
        
        # Get clusters that don't have summaries
        clusters = db.query(Cluster).all()
        logger.info(f"Found {len(clusters)} clusters to summarize")
        
        for cluster in clusters:
            # Check if summary already exists
            existing_summary = db.query(ClusterSummary).filter(ClusterSummary.cluster_id == cluster.id).first()
            if existing_summary:
                continue
            
            # Get events in this cluster
            assignments = db.query(ClusterAssignment).filter(ClusterAssignment.cluster_id == cluster.id).all()
            events = []
            
            for assignment in assignments:
                embedding = db.query(Embedding).filter(Embedding.id == assignment.embedding_id).first()
                if embedding:
                    event = db.query(TelemetryEvent).filter(TelemetryEvent.id == embedding.event_id).first()
                    if event:
                        events.append(event)
            
            if not events:
                continue
            
            # Prepare prompt for GPT-4
            events_text = []
            for event in events[:10]:  # Limit to 10 events per cluster
                skill_output_str = json.dumps(event.skill_output) if isinstance(event.skill_output, dict) else str(event.skill_output)
                events_text.append(f"- Customer: {event.customer_prompt}")
                events_text.append(f"  Error: {skill_output_str}")
                events_text.append("")
            
            prompt = f"""
Analyze these {len(events)} similar Copilot failures and provide insights:

{chr(10).join(events_text)}

Please provide:
1. Root cause analysis (2-3 sentences)
2. Common patterns identified
3. Recommended actions
4. Impact assessment

Format as JSON with keys: root_cause, patterns, recommendations, impact
"""
            
            try:
                # Call GPT-4
                logger.info(f"Generating summary for cluster {cluster.cluster_label} ({len(events)} events)")
                
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.1
                )
                
                summary_text = response.choices[0].message.content or ""
                
                # Try to parse as JSON
                try:
                    summary_json = json.loads(summary_text)
                except:
                    # If not JSON, create structured response
                    summary_json = {
                        "root_cause": summary_text[:200] + "..." if summary_text else "Unknown error pattern",
                        "patterns": f"Cluster of {len(events)} similar failures",
                        "recommendations": "Investigate common error patterns",
                        "impact": f"Affects {len(events)} user interactions"
                    }
                
                # Save summary
                cluster_summary = ClusterSummary(
                    cluster_id=cluster.id,
                    summary=summary_json,
                    root_cause=summary_json.get('root_cause', 'Unknown'),
                    recommendations=summary_json.get('recommendations', 'No recommendations'),
                    created_at=datetime.utcnow()
                )
                db.add(cluster_summary)
                db.commit()
                
                logger.info(f"✅ Generated summary for cluster {cluster.cluster_label}")
                
            except Exception as e:
                logger.error(f"❌ Error generating summary for cluster {cluster.cluster_label}: {e}")
                continue
        
        db.close()
        logger.info("✅ Completed AI summarization")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in summarization: {e}")
        return False

def run_full_analysis():
    """Run the complete AI analysis pipeline"""
    logger.info("🚀 Starting Copilot Failure AI Analysis")
    logger.info("=" * 60)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        logger.error(f"❌ Missing dependencies: {', '.join(missing)}")
        logger.info("Install with: pip install " + " ".join(missing))
        return False
    
    # Check API key
    from config import config
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "sk-your-actual-openai-key-here":
        logger.warning("⚠️  No OpenAI API key configured. Set with: $env:OPENAI_API_KEY = 'your-key'")
        logger.info("Will run embeddings and clustering without AI summaries")
    
    success_count = 0
    
    # Step 1: Generate embeddings
    logger.info("\n🧠 Step 1: Generating embeddings...")
    if generate_embeddings():
        success_count += 1
        logger.info("✅ Embeddings completed")
    else:
        logger.error("❌ Embeddings failed")
    
    # Step 2: Cluster embeddings
    logger.info("\n🎯 Step 2: Clustering embeddings...")
    if cluster_embeddings():
        success_count += 1
        logger.info("✅ Clustering completed")
    else:
        logger.error("❌ Clustering failed")
    
    # Step 3: Generate AI summaries
    logger.info("\n📝 Step 3: Generating AI summaries...")
    if generate_summaries():
        success_count += 1
        logger.info("✅ AI summaries completed")
    else:
        logger.warning("⚠️  AI summaries skipped (no API key or error)")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"📊 ANALYSIS COMPLETE: {success_count}/3 steps successful")
    logger.info("=" * 60)
    
    if success_count >= 2:
        logger.info("🎉 Analysis successful! View results at:")
        logger.info("   🌐 Web Dashboard: http://localhost:8080")
        logger.info("   📊 CLI Viewer: python view_results.py")
        logger.info("   🔗 API: python api.py (http://localhost:8000)")
    else:
        logger.error("❌ Analysis had issues. Check logs above.")
    
    return success_count >= 2

if __name__ == "__main__":
    run_full_analysis() 