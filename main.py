#!/usr/bin/env python3
"""
Copilot Failure Analysis System
Main application script for running the complete system
"""

import asyncio
import logging
import sys
from typing import List, Dict, Any
import json

from database import create_tables, get_db_session
from models import TelemetryEvent
from ingest import telemetry_ingestor
from embeddings import embedding_generator
from clustering import cluster_manager
from enhanced_clustering import enhanced_cluster_manager
from summarization import cluster_summarizer
from config import config
from excel_import import import_and_ingest_excel

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Force INFO level for better visibility
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Also add console progress indicators
def print_progress(message):
    """Print progress with timestamp"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    logger.info(message)

class CopilotFailureAnalysisSystem:
    """Main system orchestrator"""
    
    def __init__(self):
        self.initialized = False
    
    def initialize(self):
        """Initialize the system"""
        try:
            print_progress("🚀 Initializing Copilot Failure Analysis System...")
            
            # Create database tables
            print_progress("📊 Creating/verifying database tables...")
            create_tables()
            print_progress("✅ Database tables ready")
            
            self.initialized = True
            print_progress("✅ System initialization completed successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    def run_full_pipeline(self, use_openai: bool = False) -> bool:
        """Run the complete analysis pipeline"""
        try:
            if not self.initialized:
                self.initialize()
            
            print_progress("🔄 Starting full analysis pipeline...")
            
            # Step 1: Get events that need embeddings
            print_progress("📋 Checking for events that need embeddings...")
            db = get_db_session()
            try:
                from models import Embedding
                event_ids = db.query(TelemetryEvent.id).outerjoin(Embedding).filter(
                    Embedding.id.is_(None)
                ).all()
                event_ids = [id[0] for id in event_ids]
            finally:
                db.close()
            
            if not event_ids:
                print_progress("⚠️  No events found that need processing")
                return False
            
            # Step 2: Generate embeddings
            print_progress(f"🧠 Generating embeddings for {len(event_ids)} events... (This may take a few minutes)")
            embedding_count = embedding_generator.generate_embeddings_batch(
                event_ids, use_openai=use_openai
            )
            
            if embedding_count == 0:
                print_progress("❌ Failed to generate embeddings")
                return False
            
            print_progress(f"✅ Generated {embedding_count} embeddings")
            
            # Step 3: Perform enhanced hybrid clustering
            print_progress("🎯 Performing enhanced hybrid clustering with business taxonomy... (This may take a minute)")
            clustering_success = enhanced_cluster_manager.perform_enhanced_clustering()
            
            if not clustering_success:
                print_progress("❌ Enhanced clustering failed")
                return False
            
            print_progress("✅ Enhanced business-actionable clustering completed")
            
            # Step 4: Generate summaries
            print_progress("📝 Generating AI summaries for clusters... (This may take a few minutes)")
            summary_count = cluster_summarizer.summarize_all_clusters()
            
            print_progress(f"🎉 Pipeline completed successfully! Generated summaries for {summary_count} clusters")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return False
    
    def get_cluster_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all clusters"""
        try:
            return cluster_summarizer.get_top_clusters(limit=50)
        except Exception as e:
            logger.error(f"Error getting cluster summary: {e}")
            return []
    
    def print_cluster_results(self):
        """Print cluster analysis results"""
        try:
            clusters = self.get_cluster_summary()
            
            if not clusters:
                print("No clusters found")
                return
            
            print("\n" + "="*80)
            print("COPILOT FAILURE ANALYSIS RESULTS")
            print("="*80)
            
            for i, cluster in enumerate(clusters, 1):
                print(f"\n--- CLUSTER {i} (ID: {cluster['cluster_id']}) ---")
                print(f"Size: {cluster['size']} failures")
                print(f"Algorithm: {cluster['algorithm']}")
                
                if cluster['summary']:
                    print(f"\nSummary: {cluster['summary']}")
                
                if cluster['root_cause']:
                    print(f"Root Cause: {cluster['root_cause']}")
                
                if cluster['common_plugins']:
                    print(f"Common Plugins: {', '.join(cluster['common_plugins'][:3])}")
                
                if cluster['common_endpoints']:
                    print(f"Common Endpoints: {', '.join(cluster['common_endpoints'][:3])}")
                
                if cluster['sample_prompts']:
                    print(f"Sample Prompt: {cluster['sample_prompts'][0][:100]}...")
                
                print("-" * 60)
            
            print(f"\nTotal clusters analyzed: {len(clusters)}")
            
        except Exception as e:
            logger.error(f"Error printing results: {e}")

def main():
    """Main entry point"""
    system = CopilotFailureAnalysisSystem()
    
    try:
        # Initialize system
        system.initialize()
        
        # Always ingest real data from sample_telemetry_template.xlsx
        print_progress("📥 Ingesting data from sample_telemetry_template.xlsx...")
        excel_success = import_and_ingest_excel("sample_telemetry_template.xlsx")
        if not excel_success:
            print_progress("❌ Failed to ingest data from sample_telemetry_template.xlsx")
            sys.exit(1)
        print_progress("✅ Data ingestion completed")
        
        # Check if we should use OpenAI
        use_openai = "--openai" in sys.argv
        if use_openai and not config.OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured, falling back to sentence transformers")
            use_openai = False
        
        # Run the pipeline
        success = system.run_full_pipeline(use_openai=use_openai)
        
        if success:
            # Print results
            system.print_cluster_results()
            
            print("\n" + "="*80)
            print("SYSTEM READY")
            print("="*80)
            print("To start the API server, run:")
            print("python api.py")
            print("\nOr run with uvicorn:")
            print("uvicorn api:app --host 0.0.0.0 --port 8000")
            print("="*80)
        else:
            logger.error("Pipeline execution failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 